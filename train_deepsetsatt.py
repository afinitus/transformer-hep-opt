import os
import numpy as np
import h5py
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import random
import string
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from tensorflow.keras import Input, layers, models

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

def log_momenta(data):
    pt = data[:, :, 0]
    eta = data[:, :, 1]
    phi = data[:, :, 2]

    # Change the momenta of each particle to log(1 + pt)
    pt_log = np.log(1 + pt)

    # Calculate jet axis (average eta and phi of the jet)
    eta_jet = np.sum(eta * pt, axis=1) / np.sum(pt, axis=1)
    phi_jet = np.arctan2(np.sum(np.sin(phi) * pt, axis=1), np.sum(np.cos(phi) * pt, axis=1))

    # Center the rapidities and azimuthal angles
    eta_centered = eta - eta_jet[:, np.newaxis]
    phi_centered = phi - phi_jet[:, np.newaxis]

    # Ensure phi is within -pi to pi range
    phi_centered = (phi_centered + np.pi) % (2 * np.pi) - np.pi

    data_log_momenta = np.stack((pt_log, eta_centered, phi_centered), axis=-1)
    return data_log_momenta

def preprocess_data(bg_file, sig_file, num_events, num_const):
    bg = load_data(bg_file, num_events, num_const)
    sig = load_data(sig_file, num_events, num_const)

    print(f"Using bg {bg.shape} from {bg_file} and sig {sig.shape} from {sig_file}")

    dat = np.concatenate((bg, sig), 0)
    lab = np.append(np.zeros(len(bg)), np.ones(len(sig)))

    dat_log_momenta = log_momenta(dat)

    idx = np.random.permutation(len(dat))
    dat_log_momenta = dat_log_momenta[idx]
    lab = lab[idx]

    train_size = int(0.9 * len(dat))
    val_size = len(dat) - train_size

    train_data = (dat_log_momenta[:train_size], lab[:train_size])
    val_data = (dat_log_momenta[train_size:], lab[train_size:])

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


print('Loaded Train/Val Data')

# Define the model architecture as described
def DeepSetsAttClass(num_feat, num_heads=4, num_transformer=4, projection_dim=32):
    inputs = Input((None, num_feat))
    masked_inputs = layers.Masking(mask_value=0.0, name='Mask')(inputs)

    masked_features = layers.TimeDistributed(layers.Dense(projection_dim, activation=None))(masked_inputs)
    
    tdd = layers.TimeDistributed(layers.Dense(projection_dim, activation=None))(masked_features)
    tdd = layers.TimeDistributed(layers.LeakyReLU(alpha=0.01))(tdd)
    encoded_patches = layers.TimeDistributed(layers.Dense(projection_dim))(tdd)

    for _ in range(num_transformer):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim//num_heads, dropout=0.1)(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])

        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)                
        x3 = layers.Dense(4*projection_dim, activation="gelu")(x3)
        x3 = layers.Dense(projection_dim, activation="gelu")(x3)
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    pooled = layers.GlobalAvgPool1D()(representation)
    representation = layers.Dense(2*projection_dim, activation=None)(pooled)
    representation = layers.Dropout(0.1)(representation)
    representation = layers.LeakyReLU(alpha=0.01)(representation)
    
    outputs = layers.Dense(1, activation='sigmoid')(representation)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model


print('Setting up Model')
num_feat = 3
model = DeepSetsAttClass(num_feat=num_feat)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(
    monitor='val_loss',  
    patience=10,         
    verbose=1,          
    mode='min',       
    restore_best_weights=True  
)

model_checkpoint = ModelCheckpoint(
    'deepsets_checkpoints/model_deepsets_training_ckpt.weights.h5',  
    save_weights_only=True,                 
    save_freq='epoch',  # Use save_freq instead of period
    verbose=1                            
)
print('Model Setup Complete')
history = model.fit(
    train_data_np, train_labels_np,
    validation_data=(val_data_np, val_labels_np),
    epochs=128, 
    batch_size=32,
    callbacks=[early_stopping, model_checkpoint]
)

model.save('deepsets_checkpoints/model_deepsets_best_ckpt.weights.h5')


#########################################################################################################################

def get_test_data(bg_file, sig_file, num_events, num_const):
    bg = load_data(bg_file, num_events, num_const)
    sig = load_data(sig_file, num_events, num_const)

    print(f"Using bg {bg.shape} from {bg_file} and sig {sig.shape} from {sig_file}")

    dat = np.concatenate((bg, sig), 0)
    lab = np.append(np.zeros(len(bg)), np.ones(len(sig)))

    dat_log_momenta = log_momenta(dat)

    idx = np.random.permutation(len(dat))
    dat_log_momenta = dat_log_momenta[idx]
    lab = lab[idx]

    test_data = (dat_log_momenta, lab)
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
    plt.savefig(model_dir + '/dsapre_roc_test.png')
    return fpr, tpr, roc_auc

def save_roc_data(fpr, tpr, roc_auc, model_dir):
    np.savez(os.path.join(model_dir, 'dsapre_roc_data_test.npz'),
             fpr=fpr, tpr=tpr, roc_auc=roc_auc)

def save_auc_score(model_dir, auc_score):
    with open(model_dir + '/dsapre_auc.txt', 'w') as file:
        file.write(f'dsapre_auc_score\n{auc_score}\n')

model_dir = 'roc_info'
data_path_1 = '/pscratch/sd/n/nishank/humberto/FirstTime_topvsqcd_100const/top/discrete/samples_samples_nsamples200000_trunc_5000.h5'
data_path_2 = '/pscratch/sd/n/nishank/humberto/FirstTime_topvsqcd_100const/qcd/discrete/samples__nsamples200000_trunc_5000.h5'
num_const = 100

num_features = 3

print(f"Loading test set")
test_data = get_test_data(data_path_2, data_path_1, 200000, num_const)
test_data_np, test_labels_np = test_data

print("Test data shape:", test_data_np.shape)
print("Test labels shape:", test_labels_np.shape)

predictions = model.predict(test_data_np)
auc_score = roc_auc_score(test_labels_np, predictions)

# Plot and save ROC curve
fpr, tpr, roc_auc = plot_roc_curve(test_labels_np, predictions, model_dir)
save_roc_data(fpr, tpr, roc_auc, model_dir)
save_auc_score(model_dir, auc_score)

print(f"AUC Score: {auc_score:.5f}")
