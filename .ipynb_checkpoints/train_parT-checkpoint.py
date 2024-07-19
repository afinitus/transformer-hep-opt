import os
import numpy as np
import h5py
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import random
import string
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from weaver.nn.model.ParticleTransformer import ParticleTransformer
from weaver.utils.logger import _logger

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

def normalize_data(data):
    pt = data[:, :, 0]
    eta = data[:, :, 1]
    phi = data[:, :, 2]

    # Calculate total pt for normalization
    total_pt = np.sum(pt, axis=1, keepdims=True)
    pt_normalized = np.log(pt / total_pt)  # log(pt/total pt)

    # Calculate jet axis (average eta and phi of the jet)
    eta_jet = np.sum(eta * pt, axis=1) / total_pt[:, 0]
    phi_jet = np.arctan2(np.sum(np.sin(phi) * pt, axis=1), np.sum(np.cos(phi) * pt, axis=1))

    # Center the rapidities and azimuthal angles
    eta_centered = eta - eta_jet[:, np.newaxis]
    phi_centered = phi - phi_jet[:, np.newaxis]

    # Ensure phi is within -pi to pi range
    phi_centered = (phi_centered + np.pi) % (2 * np.pi) - np.pi

    data_normalized = np.stack((pt_normalized, eta_centered, phi_centered), axis=-1)
    return data_normalized

def preprocess_data(bg_file, sig_file, num_events, num_const):
    bg = load_data(bg_file, num_events, num_const)
    sig = load_data(sig_file, num_events, num_const)

    print(f"Using bg {bg.shape} from {bg_file} and sig {sig.shape} from {sig_file}")

    dat = np.concatenate((bg, sig), 0)
    lab = np.append(np.zeros(len(bg)), np.ones(len(sig)))

    dat_normalized = normalize_data(dat)

    idx = np.random.permutation(len(dat))
    dat_normalized = dat_normalized[idx]
    lab = lab[idx]

    train_size = int(0.9 * len(dat))
    val_size = len(dat) - train_size

    train_data = (dat_normalized[:train_size], lab[:train_size])
    val_data = (dat_normalized[train_size:], lab[train_size:])

    return train_data, val_data

def tensor_to_numpy(tensor_dataset):
    data_list = []
    label_list = []
    for data, label in zip(*tensor_dataset):
        data_list.append(data)
        label_list.append(label)
    data_array = np.array(data_list)
    label_array = np.array(label_list)
    return data_array, label_array

# Parameters
main_dir_discrete = '/pscratch/sd/n/nishank/humberto/FirstTime_topvsqcd_100const/'
sig_list = ['top/discrete/samples_samples_nsamples1000000_trunc_5000.h5']
bg_list = ['qcd/discrete/samples_samples_nsamples1000000_trunc_5000.h5']
num_epochs_list = [1]
dropout_list = [0.0]
num_heads_list = [4]
num_layers_list = [8]
hidden_dim_list = [256]
batch_size_list = [100]
num_events_list = [1000000]
num_const_list = [100]
lr_list = [0.001]

tag_of_train = 'top_vs_qcd_transformerdata_classifier_test_2'
log_dir = '/pscratch/sd/n/nishank/humberto/log_dir/' + tag_of_train

# Sample parameters for this run
sig = sig_list[0]
bg = bg_list[0]
num_events = num_events_list[0]
num_const = num_const_list[0]
batch_size = batch_size_list[0]

sig_path = main_dir_discrete + sig
bg_path = main_dir_discrete + bg

# Generate random suffix for this run
name_sufix = random_string()

# Load the data
train_data, val_data = preprocess_data(bg_path, sig_path, num_events, num_const)

train_data_np, train_labels_np = train_data
val_data_np, val_labels_np = val_data

# Print shapes to confirm
print("Train data shape:", train_data_np.shape)
print("Train labels shape:", train_labels_np.shape)
print("Validation data shape:", val_data_np.shape)
print("Validation labels shape:", val_labels_np.shape)

class ParticleTransformerWrapper(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.mod = ParticleTransformer(**kwargs)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token', }

    def forward(self, points, features, lorentz_vectors, mask):
        return self.mod(features, v=lorentz_vectors, mask=mask)

def get_model(data_config, **kwargs):
    cfg = dict(
        input_dim=len(data_config['input_dicts']['pf_features']),
        num_classes=len(data_config['label_value']),
        pair_input_dim=4,
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
    'input_dicts': {'pf_features': range(3)},
    'label_value': range(2),
    'input_names': ['pf_features'],
    'input_shapes': {'pf_features': (None, 100, 3)}
}

print('Setting up Particle Transformer model...')
model, model_info = get_model(data_config, input_dim=train_data_np.shape[-1], num_classes=2)

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Prepare the data loaders
train_dataset = TensorDataset(torch.tensor(train_data_np).float(), torch.tensor(train_labels_np).long())
val_dataset = TensorDataset(torch.tensor(val_data_np).float(), torch.tensor(val_labels_np).long())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Training loop
num_epochs = 128
model.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        # Calculate accuracy
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%')

# Save the trained model
torch.save(model.state_dict(), 'particle_transformer_checkpoints/model_particle_transformer_best_ckpt.pth')

#########################################################################################################################

def get_test_data(bg_file, sig_file, num_events, num_const):
    bg = load_data(bg_file, num_events, num_const)
    sig = load_data(sig_file, num_events, num_const)

    print(f"Using bg {bg.shape} from {bg_file} and sig {sig.shape} from {sig_file}")

    dat = np.concatenate((bg, sig), 0)
    lab = np.append(np.zeros(len(bg)), np.ones(len(sig)))

    dat_normalized = normalize_data(dat)

    idx = np.random.permutation(len(dat))
    dat_normalized = dat_normalized[idx]
    lab = lab[idx]

    test_data = (dat_normalized, lab)
    return test_data

def plot_roc_curve(y_true, y_score, model_dir):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (AUC = %0.5f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.yscale('log')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(model_dir + '/particle_transformer_roc_test.png')
    return fpr, tpr, roc_auc

def save_roc_data(fpr, tpr, roc_auc, model_dir):
    np.savez(os.path.join(model_dir, 'particle_transformer_roc_data_test.npz'),
             fpr=fpr, tpr=tpr, roc_auc=roc_auc)

def save_auc_score(model_dir, auc_score):
    with open(model_dir + '/particle_transformer_auc.txt', 'w') as file:
        file.write(f'particle_transformer_auc_score\n{auc_score}\n')


model_dir = 'roc_info'
data_path_1 = '/pscratch/sd/n/nishank/humberto/FirstTime_topvsqcd_100const/top/discrete/samples_samples_nsamples200000_trunc_5000.h5'
data_path_2 = '/pscratch/sd/n/nishank/humberto/FirstTime_topvsqcd_100const/qcd/discrete/samples__nsamples200000_trunc_5000.h5'
num_const = 100

print(f"Loading test set")
test_data = get_test_data(data_path_2, data_path_1, 200000, num_const)
test_data_np, test_labels_np = test_data

print("Test data shape:", test_data_np.shape)
print("Test labels shape:", test_labels_np.shape)

# Load the trained model
model.eval()
model.load_state_dict(torch.load('particle_transformer_checkpoints/model_particle_transformer_best_ckpt.pth'))

# Evaluate model
with torch.no_grad():
    test_loader = DataLoader(TensorDataset(torch.tensor(test_data_np).float(), torch.tensor(test_labels_np).long()), batch_size=batch_size, shuffle=False)
    predictions = []
    for inputs, _ in test_loader:
        outputs = model(inputs)
        predictions.extend(outputs.cpu().numpy())

predictions = np.array(predictions).flatten()
auc_score = roc_auc_score(test_labels_np, predictions)

# Plot and save ROC curve
fpr, tpr, roc_auc = plot_roc_curve(test_labels_np, predictions, model_dir)
save_roc_data(fpr, tpr, roc_auc, model_dir)
save_auc_score(model_dir, auc_score)

print(f"AUC Score: {auc_score:.5f}")
