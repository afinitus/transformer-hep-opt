import os
import numpy as np
import h5py
import random
import string
from energyflow.archs import EFN
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
    
    # **zi**: Normalized transverse momentum (pT_part / pT_jet)
    # `points[:,:,2]` is log(1 - pT_part / pT_jet), so we compute 1 - exp(points[:,:,2])
    zi = 1.0 - np.exp(data[:, :, 2])  # This gives pT_part / pT_jet
    
    # **p_hat_i**: Relative pseudorapidity and relative phi
    # Relative pseudorapidity: data[:, :, 0] (difference in rapidity between particle and jet)
    # Relative phi: data[:, :, 1] (difference in phi between particle and jet)
    p_hat_i = data[:, :, [0, 1]]  # Relative pseudorapidity and relative phi
    
    # Normalize zi (optional, but often helpful)
    #zi = zi / np.sum(zi, axis=1, keepdims=True)
    
    return zi, p_hat_i, pid

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
train_zi, train_p_hat_i, train_labels = get_data(train_file)
print('Loading validation data...')
val_zi, val_p_hat_i, val_labels = get_data(val_file)

# Print shapes to confirm
print("Train zi shape:", train_zi.shape)
print("Train p_hat_i shape:", train_p_hat_i.shape)
print("Train labels shape:", train_labels.shape)
print("Validation zi shape:", val_zi.shape)
print("Validation p_hat_i shape:", val_p_hat_i.shape)
print("Validation labels shape:", val_labels.shape)

print('Setting up EFN model...')
input_dim = train_p_hat_i.shape[-1]

# Creating the EFN model with hyperparameters
model = EFN(
    input_dim=input_dim,
    Phi_sizes=(128, 128, 256),
    F_sizes=(128, 128, 128),
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'],
    output_dim=1,
    output_act='sigmoid',
    filepath='efn_checkpoints/model_efn_training_ckpt_epoch_{epoch:02d}.weights.h5',
    modelcheck_opts={'save_best_only':True, 'verbose':1, 'save_weights_only':True},
    patience=10,
    earlystop_opts={'restore_best_weights':True, 'verbose':1},
    save_while_training=True,
    compile=True,
    summary=True,
)

history = model.fit(
    x=[train_zi, train_p_hat_i],
    y=train_labels,
    validation_data=([val_zi, val_p_hat_i], val_labels),
    epochs=num_epochs,
    batch_size=batch_size
)

model.save('efn_checkpoints/model_efn_best_ckpt.h5')

# Test set evaluation
print('Loading test data...')
test_zi, test_p_hat_i, test_labels = get_data(test_file)

print("Test zi shape:", test_zi.shape)
print("Test p_hat_i shape:", test_p_hat_i.shape)
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
    plt.savefig(os.path.join(model_dir, 'efn_roc_test.png'))
    return fpr, tpr, roc_auc

def save_roc_data(fpr, tpr, roc_auc, model_dir):
    np.savez(os.path.join(model_dir, 'efn_roc_data_test.npz'),
             fpr=fpr, tpr=tpr, roc_auc=roc_auc)

def save_auc_score(model_dir, auc_score):
    with open(os.path.join(model_dir, 'efn_auc.txt'), 'w') as file:
        file.write(f'efn_auc_score\n{auc_score}\n')

model_dir = 'roc_info'
os.makedirs(model_dir, exist_ok=True)

predictions = model.predict([test_zi, test_p_hat_i])
auc_score = roc_auc_score(test_labels, predictions)

# Plot and save ROC curve
fpr, tpr, roc_auc = plot_roc_curve(test_labels, predictions, model_dir)
save_roc_data(fpr, tpr, roc_auc, model_dir)
save_auc_score(model_dir, auc_score)

print(f"AUC Score: {auc_score:.9f}")
