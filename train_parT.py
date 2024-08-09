import os
import numpy as np
import h5py
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import random
import string
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from weaver.nn.model.ParticleTransformer import ParticleTransformer
from weaver.utils.logger import _logger
import torch.nn.functional as F
import time

def random_string():
    N = 7
    res = ''.join(random.choices(string.ascii_uppercase + string.digits, k=N))
    return str(res)

def load_data(file, num_events, num_const):
    if file.endswith("npz"):
        dat = np.load(file)["jets"][:num_events, :num_const]
    elif file.endswith("h5"):
        dat = pd.read_hdf(file, key="discretized", stop=num_events)
        dat = dat.to_numpy(dtype=np.float32)[:, :num_const * 3]
        dat = dat.reshape(dat.shape[0], -1, 3)
    else:
        raise ValueError("Filetype not supported")
    dat = np.delete(dat, np.where(dat[:, 0, 0] == 0)[0], axis=0)
    dat[dat == -1] = 0
    return dat

def reconstruct_4_momenta(pt, eta, phi):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy = np.sqrt(px**2 + py**2 + pz**2)  # Assuming massless particles
    return np.stack((px, py, pz, energy), axis=-1)

def normalize_data(data):
    pt = data[:, :, 0]
    eta = data[:, :, 1]
    phi = data[:, :, 2]

    total_pt = np.sum(pt, axis=1, keepdims=True)
    total_pt[total_pt == 0] = 1e-9  # Prevent division by zero

    pt_normalized = np.log(pt / total_pt)
    pt_normalized[np.isneginf(pt_normalized)] = 0  # Replace -inf with 0

    eta_jet = np.sum(eta * pt, axis=1) / total_pt[:, 0]
    phi_jet = np.arctan2(np.sum(np.sin(phi) * pt, axis=1), np.sum(np.cos(phi) * pt, axis=1))

    eta_centered = eta - eta_jet[:, np.newaxis]
    phi_centered = phi - phi_jet[:, np.newaxis]
    phi_centered = (phi_centered + np.pi) % (2 * np.pi) - np.pi

    features_normalized = np.stack((pt_normalized, eta_centered, phi_centered), axis=-1)
    lorentz_vectors = reconstruct_4_momenta(pt, eta, phi)
    return features_normalized, lorentz_vectors

class ParticleTransformerDataset(Dataset):
    def __init__(self, bg_file, sig_file, num_events, num_const):
        bg = load_data(bg_file, num_events, num_const)
        sig = load_data(sig_file, num_events, num_const)

        print(f"Using bg {bg.shape} from {bg_file} and sig {sig.shape} from {sig_file}")

        dat = np.concatenate((bg, sig), 0)
        lab = np.append(np.zeros(len(bg)), np.ones(len(sig)))

        features_normalized, lorentz_vectors = normalize_data(dat)

        idx = np.random.permutation(len(dat))
        features_normalized = features_normalized[idx]
        lorentz_vectors = lorentz_vectors[idx]
        lab = lab[idx]

        self.features = torch.tensor(features_normalized).float().permute(0, 2, 1)
        self.lorentz_vectors = torch.tensor(lorentz_vectors).float().permute(0, 2, 1)
        self.labels = torch.tensor(lab).long()
        self.num_samples = len(lab)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        num_cands = self.features.shape[2]
        mask = torch.ones(1, num_cands, dtype=torch.bool)  # Ensure the mask is of shape (1, num_cands)
        return {
            "x": self.features[idx],
            "v": self.lorentz_vectors[idx],
            "mask": mask
        }, self.labels[idx]

class ParticleTransformerWrapper(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.mod = ParticleTransformer(**kwargs)
        self.input_bn = torch.nn.BatchNorm1d(3)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token', }

    def forward(self, points, features, lorentz_vectors, mask):
        features = self.input_bn(features)
        return self.mod(features, v=lorentz_vectors, mask=mask)

def get_model(data_config, **kwargs):
    cfg = dict(
        input_dim=3,
        num_classes=len(data_config['label_value']),
        pair_input_dim=4,
        use_pre_activation_pair=False,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],
        num_heads=10,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        fc_params=[],
        activation='gelu',
        trim=True,
        for_inference=False,
    )
    cfg.update(**kwargs)
    _logger.info('Model config: %s' % str(cfg))

    model = ParticleTransformerWrapper(**cfg)

    model_info = {
        'input_names': list(data_config['input_names']),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config['input_shapes'].items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config['input_names']}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info

data_config = {
    'input_dicts': {'pf_features': range(3)},
    'label_value': range(2),
    'input_names': ['pf_features'],
    'input_shapes': {'pf_features': (None, 100, 3)}
}

print('Setting up Particle Transformer model...')
model, model_info = get_model(data_config, input_dim=3, num_classes=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("CUDA available:", torch.cuda.is_available())
model = torch.nn.DataParallel(model)
model = model.to(device)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

bg_file = '/pscratch/sd/n/nishank/humberto/FirstTime_topvsqcd_100const/qcd/discrete/samples_samples_nsamples1000000_trunc_5000.h5'
sig_file = '/pscratch/sd/n/nishank/humberto/FirstTime_topvsqcd_100const/top/discrete/samples_samples_nsamples1000000_trunc_5000.h5'

train_dataset = ParticleTransformerDataset(bg_file, sig_file, num_events=1000000, num_const=100)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

num_epochs = 128
model.train()

def one_hot(labels, num_classes):
    return F.one_hot(labels, num_classes=num_classes).float()

for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    epoch_start_time = time.time()
    for batch_idx, batch in enumerate(train_loader):
        #if batch_idx == 200:
        #    break
        batch_start_time = time.time()
        inputs, labels = batch
        features = inputs["x"].to(device)
        lorentz_vectors = inputs["v"].to(device)
        mask = inputs["mask"].to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features, features, lorentz_vectors, mask)
        if torch.isnan(outputs).any():
            print(f'NaN detected in outputs at step {batch_idx}')
            continue
        labels_one_hot = one_hot(labels, num_classes=2)
        loss = criterion(outputs, labels_one_hot)
        if torch.isnan(loss).any():
            print(f'NaN detected in loss at step {batch_idx}')
            continue
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted.argmax(dim=1) == labels).sum().item()
        
        if batch_idx % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    epoch_end_time = time.time()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%, Time: {epoch_end_time - epoch_start_time:.2f}s')

torch.save(model.state_dict(), 'particle_transformer_checkpoints/model_particle_transformer_best_ckpt.pth')

# Testing phase
print(f"Loading test set")
data_path_1 = '/pscratch/sd/n/nishank/humberto/FirstTime_topvsqcd_100const/top/discrete/samples_samples_nsamples200000_trunc_5000.h5'
data_path_2 = '/pscratch/sd/n/nishank/humberto/FirstTime_topvsqcd_100const/qcd/discrete/samples__nsamples200000_trunc_5000.h5'

test_dataset = ParticleTransformerDataset(data_path_2, data_path_1, num_events=200000, num_const=100)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

model.eval()
model.load_state_dict(torch.load('particle_transformer_checkpoints/model_particle_transformer_best_ckpt.pth'))

predictions = []
test_labels = []

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        inputs, labels = batch
        features = inputs["x"].to(device)
        lorentz_vectors = inputs["v"].to(device)
        mask = inputs["mask"].to(device)
        
        outputs = model(features, features, lorentz_vectors, mask)
        if torch.isnan(outputs).any():
            print(f'NaN detected in outputs at step {batch_idx}')
            continue

        # Assuming outputs are logits for two classes, take the second column for the positive class prediction
        preds = torch.sigmoid(outputs)[:, 1].cpu().numpy()

        predictions.extend(preds)
        test_labels.extend(labels.cpu().numpy())
        
        if batch_idx % 10 == 0:
            print(f'Step [{batch_idx+1}/{len(test_loader)}]')

predictions = np.array(predictions)
test_labels = np.array(test_labels)

# Check shapes before calculating AUC score
print(f'Predictions shape: {predictions.shape}')
print(f'Test labels shape: {test_labels.shape}')

# Ensure predictions and test_labels have consistent lengths
if len(predictions) != len(test_labels):
    raise ValueError(f"Inconsistent number of samples: predictions={len(predictions)}, test_labels={len(test_labels)}")

auc_score = roc_auc_score(test_labels, predictions)

def plot_roc_curve(y_true, y_score, model_dir):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(tpr, 1/fpr, color='darkorange', lw=lw, label='ROC curve (AUC = %0.5f)' % roc_auc)
    #plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.yscale('log')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(model_dir, 'particle_transformer_roc_test.png'))
    return fpr, tpr, roc_auc

def save_roc_data(fpr, tpr, roc_auc, model_dir):
    np.savez(os.path.join(model_dir, 'particle_transformer_roc_data_test.npz'), fpr=fpr, tpr=tpr, roc_auc=roc_auc)

def save_auc_score(model_dir, auc_score):
    with open(os.path.join(model_dir, 'particle_transformer_auc.txt'), 'w') as file:
        file.write(f'particle_transformer_auc_score\n{auc_score}\n')

model_dir = 'roc_info'
fpr, tpr, roc_auc = plot_roc_curve(test_labels, predictions, model_dir)
save_roc_data(fpr, tpr, roc_auc, model_dir)
save_auc_score(model_dir, auc_score)

print(f"AUC Score: {auc_score:.5f}")
