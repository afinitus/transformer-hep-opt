import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
import random
import string
from energyflow.archs import EFN
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

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

def get_data(bg_file, sig_file, num_events, num_const):
    bg = load_data(bg_file, num_events, num_const)
    sig = load_data(sig_file, num_events, num_const)

    print(f"Using bg {bg.shape} from {bg_file} and sig {sig.shape} from {sig_file}")

    dat = np.concatenate((bg, sig), 0)
    lab = np.append(np.zeros(len(bg)), np.ones(len(sig)))

    idx = np.random.permutation(len(dat))
    dat = dat[idx]
    lab = lab[idx]

    pt = dat[:, :, 0]
    angular_info = dat[:, :, 1:]

    train_size = int(0.9 * len(dat))
    val_size = len(dat) - train_size

    train_data = (pt[:train_size], angular_info[:train_size], lab[:train_size])
    val_data = (pt[train_size:], angular_info[train_size:], lab[train_size:])

    return train_data, val_data

def tensor_to_numpy(tensor_dataset):
    data_list = []
    angular_list = []
    label_list = []
    for data, angular, label in zip(*tensor_dataset):
        data_list.append(data)
        angular_list.append(angular)
        label_list.append(label)
    data_array = np.array(data_list)
    angular_array = np.array(angular_list)
    label_array = np.array(label_list)
    return data_array, angular_array, label_array

# Parameters
main_dir_discrete = '/pscratch/sd/n/nishank/humberto/FirstTime_topvsqcd_100const/'
sig_list = ['top/discrete/samples_samples_nsamples1000000_trunc_5000.h5']
bg_list = ['qcd/discrete/samples_samples_nsamples1000000_trunc_5000.h5']
num_epochs_list = [128]
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
train_data, val_data = get_data(bg_path, sig_path, num_events, num_const)

train_pt_np, train_angular_np, train_labels_np = train_data
val_pt_np, val_angular_np, val_labels_np = val_data

# Print shapes to confirm
print("Train pt shape:", train_pt_np.shape)
print("Train angular shape:", train_angular_np.shape)
print("Train labels shape:", train_labels_np.shape)
print("Validation pt shape:", val_pt_np.shape)
print("Validation angular shape:", val_angular_np.shape)
print("Validation labels shape:", val_labels_np.shape)

print('Setting up EFN model...')
input_dim = train_angular_np.shape[-1]

# Creating the EFN model with hyperparameters
model = EFN(
    input_dim=input_dim,
    Phi_sizes=(100, 100, 256),
    F_sizes=(100, 100, 100),
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'],
    output_dim=1,  # Setting the output dimension to 1
    output_act='sigmoid',  # Setting the output activation to sigmoid
    filepath='efn_checkpoints/model_efn_training_ckpt_epoch_{epoch:02d}.weights.h5',
    modelcheck_opts={'save_best_only':True, 'verbose':1, 'save_weights_only':True},
    earlystop_opts={'restore_best_weights':True, 'verbose':1, 'patience':10},
    save_while_training=True,
    compile=True,  # Ensure the model is compiled
    summary=True,  # Print a summary of the model
)

history = model.fit(
    x=[train_pt_np, train_angular_np],
    y=train_labels_np,
    validation_data=([val_pt_np, val_angular_np], val_labels_np),
    epochs=num_epochs_list[0],
    batch_size=batch_size
)

model.save('efn_checkpoints/model_efn_best_ckpt.h5')


#########################################################################################################################

def get_test_data(bg_file, sig_file, num_events, num_const):
    bg = load_data(bg_file, num_events, num_const)
    sig = load_data(sig_file, num_events, num_const)

    print(f"Using bg {bg.shape} from {bg_file} and sig {sig.shape} from {sig_file}")

    dat = np.concatenate((bg, sig), 0)
    lab = np.append(np.zeros(len(bg)), np.ones(len(sig)))

    idx = np.random.permutation(len(dat))
    dat = dat[idx]
    lab = lab[idx]

    pt = dat[:, :, 0]
    angular_info = dat[:, :, 1:]

    test_data = (pt, angular_info, lab)
    return test_data

def plot_roc_curve(y_true, y_score, model_dir):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    lw = 2
    plt.plot(tpr, 1/fpr, color='darkorange', lw=lw, label='ROC curve (AUC = %0.5f)' % roc_auc)
    #plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.yscale('log')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(model_dir + '/efn_roc_test.png')
    return fpr, tpr, roc_auc

def save_roc_data(fpr, tpr, roc_auc, model_dir):
    np.savez(os.path.join(model_dir, 'efn_roc_data_test.npz'),
             fpr=fpr, tpr=tpr, roc_auc=roc_auc)

def save_auc_score(model_dir, auc_score):
    with open(model_dir + '/efn_auc.txt', 'w') as file:
        file.write(f'efn_auc_score\n{auc_score}\n')


model_dir = 'roc_info'
data_path_1 = '/pscratch/sd/n/nishank/humberto/FirstTime_topvsqcd_100const/top/discrete/samples_samples_nsamples200000_trunc_5000.h5'
data_path_2 = '/pscratch/sd/n/nishank/humberto/FirstTime_topvsqcd_100const/qcd/discrete/samples__nsamples200000_trunc_5000.h5'
num_const = 100

num_features = 3

print(f"Loading test set")
test_data = get_test_data(data_path_2, data_path_1, 200000, num_const)
test_pt_np, test_angular_np, test_labels_np = test_data

print("Test pt shape:", test_pt_np.shape)
print("Test angular shape:", test_angular_np.shape)
print("Test labels shape:", test_labels_np.shape)

predictions = model.predict([test_pt_np, test_angular_np])
auc_score = roc_auc_score(test_labels_np, predictions)

# Plot and save ROC curve
fpr, tpr, roc_auc = plot_roc_curve(test_labels_np, predictions, model_dir)
save_roc_data(fpr, tpr, roc_auc, model_dir)
save_auc_score(model_dir, auc_score)

print(f"AUC Score: {auc_score:.5f}")
