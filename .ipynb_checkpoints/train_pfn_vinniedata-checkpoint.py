import os
import numpy as np
import h5py
import random
import string
from energyflow.archs import PFN
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

def random_string():
    N = 7
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=N))

def load_data(file_path):
    with h5py.File(file_path, 'r') as h5file:
        data = h5file['data'][:]
        pid = h5file['pid'][:]
    return data, pid

def get_data(file_path):
    data, pid = load_data(file_path)
    
    # Shuffle the data
    idx = np.random.permutation(len(data))
    data = data[idx]
    pid = pid[idx]
    
    # **Extract the three features**
    # Relative pseudorapidity: data[:, :, 0]
    # Relative phi: data[:, :, 1]
    # Normalized transverse momentum (pT_part / pT_jet): 1 - exp(data[:, :, 2])
    features = np.stack([
        data[:, :, 0],  # Relative pseudorapidity
        data[:, :, 1],  # Relative phi
        1.0 - np.exp(data[:, :, 2])  # Normalized transverse momentum (pT_part / pT_jet)
    ], axis=-1)
    
    return features, pid

def check_for_nan_inf(data, name=""):
    if np.isnan(data).any():
        print(f"Warning: NaN values detected in {name}.")
    if np.isinf(data).any():
        print(f"Warning: Inf values detected in {name}.")

# Parameters
train_file = '/global/cfs/cdirs/m3246/vmikuni/for_nishank/Aachen/train_ttbar.h5'
val_file = '/global/cfs/cdirs/m3246/vmikuni/for_nishank/Aachen/val_ttbar.h5'
test_file = '/global/cfs/cdirs/m3246/vmikuni/for_nishank/Aachen/test_ttbar.h5'

num_epochs = 1
batch_size = 100
log_dir = '/pscratch/sd/n/nishank/humberto/log_dir/top_vs_qcd_transformerdata_classifier_test_2'

# Generate random suffix for this run
name_suffix = random_string()

# Load the data
print('Loading training data...')
train_data, train_labels = get_data(train_file)
check_for_nan_inf(train_data, "training data")
check_for_nan_inf(train_labels, "training labels")

# Load and check the validation data
print('Loading validation data...')
val_data, val_labels = get_data(val_file)
check_for_nan_inf(val_data, "validation data")
check_for_nan_inf(val_labels, "validation labels")

# Print shapes to confirm
print("Train data shape:", train_data.shape)
print("Train labels shape:", train_labels.shape)
print("Validation data shape:", val_data.shape)
print("Validation labels shape:", val_labels.shape)

print('Setting up PFN model...')
input_dim = train_data.shape[-1]  # Should now be 3 (pseudorapidity, phi, pT_part/pT_jet)

# Creating the PFN model with hyperparameters
model = PFN(
    input_dim=input_dim,
    Phi_sizes=(128, 128, 256),
    F_sizes=(128, 128, 128),
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'],
    output_dim=1,
    output_act='sigmoid',
    filepath='pfn_checkpoints/model_pfn_training_ckpt_epoch_{epoch:02d}.weights.h5',
    modelcheck_opts={'save_best_only':True, 'verbose':1, 'save_weights_only':True},
    earlystop_opts={'restore_best_weights':True, 'verbose':1, 'patience':10},
    save_while_training=True,
    compile=True,
    summary=True,
)

history = model.fit(
    x=train_data,
    y=train_labels,
    validation_data=(val_data, val_labels),
    epochs=num_epochs,
    batch_size=batch_size
)

model.save('pfn_checkpoints/model_pfn_best_ckpt.h5')

# Test set evaluation
print('Loading test data...')
test_data, test_labels = get_data(test_file)

print("Test data shape:", test_data.shape)
print("Test labels shape:", test_labels.shape)

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
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(model_dir, 'pfn_roc_test.png'))
    return fpr, tpr, roc_auc

def save_roc_data(fpr, tpr, roc_auc, model_dir):
    np.savez(os.path.join(model_dir, 'pfn_roc_data_test.npz'),
             fpr=fpr, tpr=tpr, roc_auc=roc_auc)

def save_auc_score(model_dir, auc_score):
    with open(os.path.join(model_dir, 'pfn_auc.txt'), 'w') as file:
        file.write(f'pfn_auc_score\n{auc_score}\n')

model_dir = 'roc_info'
os.makedirs(model_dir, exist_ok=True)

predictions = model.predict(test_data)
auc_score = roc_auc_score(test_labels, predictions)

# Plot and save ROC curve
fpr, tpr, roc_auc = plot_roc_curve(test_labels, predictions, model_dir)
save_roc_data(fpr, tpr, roc_auc, model_dir)
save_auc_score(model_dir, auc_score)

print(f"AUC Score: {auc_score:.9f}")
