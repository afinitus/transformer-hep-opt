import os
import numpy as np
import h5py
from tensorflow.keras import Input, layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import random
import string

def random_string():
    N = 7
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=N))

def load_data(file_path):
    with h5py.File(file_path, 'r') as h5file:
        data = h5file['data'][:]
        pid = h5file['pid'][:]
    return data, pid

def preprocess_data(file_path):
    data, labels = load_data(file_path)
    
    # Shuffle the data
    idx = np.random.permutation(len(data))
    data = data[idx]
    labels = labels[idx]
    
    return data, labels

import tensorflow as tf
from tensorflow.keras import layers, models

class ApplyMask(layers.Layer):
    def call(self, inputs, mask=None):
        x, m = inputs
        return x * tf.cast(m, x.dtype)[:, :, tf.newaxis]

    def compute_output_shape(self, input_shape):
        return input_shape[0]

import tensorflow as tf
from tensorflow.keras import layers, models

class CreateMask(layers.Layer):
    def call(self, inputs):
        return tf.cast(tf.not_equal(tf.reduce_sum(inputs, axis=-1), 0), tf.float32)

class ApplyMask(layers.Layer):
    def call(self, inputs):
        x, mask = inputs
        return x * mask[:, :, tf.newaxis]

class MaskedGlobalAveragePooling(layers.Layer):
    def call(self, inputs):
        x, mask = inputs
        return tf.reduce_sum(x * mask[:, :, tf.newaxis], axis=1) / tf.reduce_sum(mask, axis=1)[:, tf.newaxis]

def DeepSetsAttClass(num_feat, num_heads=4, num_transformer=4, projection_dim=32):
    inputs = layers.Input((None, num_feat))
    print(f"Input shape: {inputs.shape}")
    
    mask = CreateMask()(inputs)
    print(f"Mask shape: {mask.shape}")
    
    masked_features = layers.TimeDistributed(layers.Dense(projection_dim, activation=None))(inputs)
    masked_features = ApplyMask()([masked_features, mask])
    print(f"Masked features shape: {masked_features.shape}")
    
    tdd = layers.TimeDistributed(layers.Dense(projection_dim, activation=None))(masked_features)
    tdd = layers.TimeDistributed(layers.LeakyReLU(alpha=0.01))(tdd)
    encoded_patches = layers.TimeDistributed(layers.Dense(projection_dim))(tdd)
    encoded_patches = ApplyMask()([encoded_patches, mask])
    print(f"Encoded patches shape: {encoded_patches.shape}")
    
    for i in range(num_transformer):
        print(f"Transformer layer {i+1}")
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=projection_dim//num_heads, 
            dropout=0.1
        )(x1, x1, attention_mask=mask[:, tf.newaxis, tf.newaxis, :])
        print(f"  Attention output shape: {attention_output.shape}")
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)                
        x3 = layers.Dense(4*projection_dim, activation="gelu")(x3)
        x3 = layers.Dense(projection_dim, activation="gelu")(x3)
        encoded_patches = layers.Add()([x3, x2])
        encoded_patches = ApplyMask()([encoded_patches, mask])
        print(f"  Encoded patches shape after transformer: {encoded_patches.shape}")
    
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    pooled = MaskedGlobalAveragePooling()([representation, mask])
    print(f"Pooled shape: {pooled.shape}")
    
    representation = layers.Dense(2*projection_dim, activation=None)(pooled)
    representation = layers.Dropout(0.1)(representation)
    representation = layers.LeakyReLU(alpha=0.01)(representation)
    print(f"Final representation shape: {representation.shape}")
    
    outputs = layers.Dense(1, activation='sigmoid')(representation)
    print(f"Output shape: {outputs.shape}")
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model



# Parameters
train_file = '/global/cfs/cdirs/m3246/vmikuni/for_nishank/Aachen/train_ttbar.h5'
val_file = '/global/cfs/cdirs/m3246/vmikuni/for_nishank/Aachen/val_ttbar.h5'
test_file = '/global/cfs/cdirs/m3246/vmikuni/for_nishank/Aachen/test_ttbar.h5'

num_epochs = 128
batch_size = 32

# Generate random suffix for this run
name_suffix = random_string()

# Load the data
print('Loading training data...')
train_data, train_labels = preprocess_data(train_file)
print('Loading validation data...')
val_data, val_labels = preprocess_data(val_file)

# Print shapes to confirm
print("Train data shape:", train_data.shape)
print("Train labels shape:", train_labels.shape)
print("Validation data shape:", val_data.shape)
print("Validation labels shape:", val_labels.shape)

print('Setting up Model')
num_feat = 7  # Using all 7 features
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
    save_freq='epoch',
    verbose=1                            
)

print('Model Setup Complete')
history = model.fit(
    train_data, train_labels,
    validation_data=(val_data, val_labels),
    epochs=num_epochs, 
    batch_size=batch_size,
    callbacks=[early_stopping, model_checkpoint]
)

model.save('deepsets_checkpoints/model_deepsets_best_ckpt.weights.h5')

# Testing phase
print(f"Loading test set")
test_data, test_labels = preprocess_data(test_file)

print("Test data shape:", test_data.shape)
print("Test labels shape:", test_labels.shape)

predictions = model.predict(test_data)
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
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(model_dir, 'dsa_roc_test.png'))
    return fpr, tpr, roc_auc

def save_roc_data(fpr, tpr, roc_auc, model_dir):
    np.savez(os.path.join(model_dir, 'dsa_roc_data_test.npz'),
             fpr=fpr, tpr=tpr, roc_auc=roc_auc)

def save_auc_score(model_dir, auc_score):
    with open(os.path.join(model_dir, 'dsapre_auc.txt'), 'w') as file:
        file.write(f'dsa_auc_score\n{auc_score}\n')

model_dir = 'roc_info'
os.makedirs(model_dir, exist_ok=True)

# Plot and save ROC curve
fpr, tpr, roc_auc = plot_roc_curve(test_labels, predictions, model_dir)
save_roc_data(fpr, tpr, roc_auc, model_dir)
save_auc_score(model_dir, auc_score)

print(f"AUC Score: {auc_score:.9f}")