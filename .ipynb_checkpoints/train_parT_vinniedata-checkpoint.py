import os
import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader, Dataset
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
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=N))

def load_data(file_path):
    with h5py.File(file_path, 'r') as h5file:
        data = h5file['data'][:]
        pid = h5file['pid'][:]
    return data, pid

class ParticleTransformerDataset(Dataset):
    def __init__(self, file_path):
        data, self.labels = load_data(file_path)
        
        # Shuffle the data
        idx = np.random.permutation(len(data))
        data = data[idx]
        self.labels = self.labels[idx]

        # Split data into features
        self.features = torch.tensor(data).float().permute(0, 2, 1)
        self.labels = torch.tensor(self.labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        num_cands = self.features.shape[2]
        mask = torch.ones(1, num_cands, dtype=torch.bool)
        return {
            "x": self.features[idx],
            "mask": mask
        }, self.labels[idx]

class ParticleTransformerWrapper(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.mod = ParticleTransformer(**kwargs)
        self.input_bn = torch.nn.BatchNorm1d(7)  # Changed to 7 for all features

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token', }

    def forward(self, features, mask):
        features = self.input_bn(features)
        return self.mod(features, mask=mask)

def get_model(data_config, **kwargs):
    cfg = dict(
        input_dim=7,  # Changed to 7 for all features
        num_classes=len(data_config['label_value']),
        use_pre_activation_pair=False,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],
        num_heads=8,
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
    'input_dicts': {'pf_features': range(7)},
    'label_value': range(2),
    'input_names': ['pf_features'],
    'input_shapes': {'pf_features': (None, 100, 7)}
}

print('Setting up Particle Transformer model...')
model, model_info = get_model(data_config, input_dim=7, num_classes=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("CUDA available:", torch.cuda.is_available())
model = torch.nn.DataParallel(model)
model = model.to(device)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_file = '/global/cfs/cdirs/m3246/vmikuni/for_nishank/Aachen/train_ttbar.h5'
val_file = '/global/cfs/cdirs/m3246/vmikuni/for_nishank/Aachen/val_ttbar.h5'
test_file = '/global/cfs/cdirs/m3246/vmikuni/for_nishank/Aachen/test_ttbar.h5'

train_dataset = ParticleTransformerDataset(train_file)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

num_epochs = 50
model.train()

def one_hot(labels, num_classes):
    return F.one_hot(labels, num_classes=num_classes).float()

for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    epoch_start_time = time.time()
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        batch_start_time = time.time()
        features = inputs["x"].to(device)
        mask = inputs["mask"].to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features, mask)
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
test_dataset = ParticleTransformerDataset(test_file)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

model.eval()
model.load_state_dict(torch.load('particle_transformer_checkpoints/model_particle_transformer_best_ckpt.pth'))

predictions = []
test_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        features = inputs["x"].to(device)
        mask = inputs["mask"].to(device)
        
        outputs = model(features, mask)
        if torch.isnan(outputs).any():
            print(f'NaN detected in outputs')
            continue

        preds = torch.sigmoid(outputs)[:, 1].cpu().numpy()

        predictions.extend(preds)
        test_labels.extend(labels.cpu().numpy())

predictions = np.array(predictions)
test_labels = np.array(test_labels)

auc_score = roc_auc_score(test_labels, predictions)

def plot_roc_curve(y_true, y_score, model_dir):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(tpr, 1/fpr, color='darkorange', lw=lw, label='ROC curve (AUC = %0.9f)' % roc_auc)
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

print(f"AUC Score: {auc_score:.9f}")