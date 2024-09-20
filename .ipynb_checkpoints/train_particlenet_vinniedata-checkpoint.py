import os
import numpy as np
import h5py
import random
import string
import logging
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

# Assuming tf_keras_model is a custom module containing get_particle_net and get_particle_net_lite
from tf_keras_model import get_particle_net, get_particle_net_lite

def random_string():
    N = 7
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=N))

class Dataset(object):
    def __init__(self, filepath, pad_len=128, data_format='channel_last'):
        self.filepath = filepath
        self.pad_len = pad_len
        assert data_format in ('channel_first', 'channel_last')
        self.stack_axis = 1 if data_format == 'channel_first' else -1
        self._values = {}
        self._label = None
        self._load()

    def _load(self):
        logging.info('Start loading file %s' % self.filepath)
        with h5py.File(self.filepath, 'r') as f:
            # Load the main data
            data = f['data'][:]
            if data.shape[1] < self.pad_len:
                pad_width = ((0, 0), (0, self.pad_len - data.shape[1]), (0, 0))
                data = np.pad(data, pad_width, mode='constant')
            elif data.shape[1] > self.pad_len:
                data = data[:, :self.pad_len, :]
            
            # Split the data into points, features, and mask
            self._values['points'] = data[:, :, :2]  # First two columns
            self._values['features'] = data[:, :, 2:6]  # Next four columns
            self._values['mask'] = (data[:, :, 0] != 0).astype(np.float32)  # Use first column for mask
            
            # Load jet features
            self._values['jet_features'] = f['jet'][:]
            
            # Load labels (pid)
            self._label = f['pid'][:].astype(np.int32)
        
        logging.info('Finished loading file %s' % self.filepath)

    def __len__(self):
        return len(self._label)

    def __getitem__(self, key):
        if key == 'label':
            return self._label
        else:
            return self._values.get(key)
    
    @property
    def X(self):
        return self._values
    
    @property
    def y(self):
        return self._label

    def shuffle(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        shuffle_indices = np.arange(self.__len__())
        np.random.shuffle(shuffle_indices)
        for k in self._values:
            self._values[k] = self._values[k][shuffle_indices]
        self._label = self._label[shuffle_indices]

def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 10:
        lr *= 0.1
    elif epoch > 20:
        lr *= 0.01
    logging.info('Learning rate: %f' % lr)
    return lr

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
    plt.savefig(os.path.join(model_dir, 'particlenet_roc_test.png'))
    return fpr, tpr, roc_auc

def save_roc_data(fpr, tpr, roc_auc, model_dir):
    np.savez(os.path.join(model_dir, 'particlenet_roc_data_test.npz'),
             fpr=fpr, tpr=tpr, roc_auc=roc_auc)

def save_auc_score(model_dir, auc_score):
    with open(os.path.join(model_dir, 'particlenet_auc.txt'), 'w') as file:
        file.write(f'particlenet_auc_score\n{auc_score}\n')

# Main execution
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

    # Parameters
    train_file = '/global/cfs/cdirs/m3246/vmikuni/for_nishank/Aachen/train_ttbar.h5'
    val_file = '/global/cfs/cdirs/m3246/vmikuni/for_nishank/Aachen/val_ttbar.h5'
    test_file = '/global/cfs/cdirs/m3246/vmikuni/for_nishank/Aachen/test_ttbar.h5'
    model_type = 'particle_net'  # choose between 'particle_net' and 'particle_net_lite'
    num_epochs = 1
    batch_size = 1024 if 'lite' in model_type else 384
    log_dir = '/pscratch/sd/n/nishank/humberto/log_dir/top_vs_qcd_transformerdata_classifier_test_2'

    # Generate random suffix for this run
    name_suffix = random_string()

    # Load the data
    print('Loading training data...')
    train_dataset = Dataset(train_file, data_format='channel_last')
    print('Loading validation data...')
    val_dataset = Dataset(val_file, data_format='channel_last')

    # Print shapes to confirm
    print("Train dataset shape:")
    for k, v in train_dataset.X.items():
        print(f"  {k}: {v.shape}")
    print(f"Train labels shape: {train_dataset.y.shape}")

    print("Validation dataset shape:")
    for k, v in val_dataset.X.items():
        print(f"  {k}: {v.shape}")
    print(f"Validation labels shape: {val_dataset.y.shape}")

    print('Setting up ParticleNet model...')
    num_classes = 2  # Binary classification (0 or 1)
    input_shapes = {k: v.shape[1:] for k, v in train_dataset.X.items()}
    if 'lite' in model_type:
        model = get_particle_net_lite(num_classes, input_shapes)
    else:
        model = get_particle_net(num_classes, input_shapes)

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(learning_rate=lr_schedule(0)),
                  metrics=['accuracy'])
    model.summary()

    # Prepare model saving directory
# Prepare model saving directory
    save_dir = 'model_checkpoints'
    model_name = f'{model_type}_model.{{epoch:03d}}.keras'  # Use .keras instead of .h5
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

# Prepare callbacks
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=filepath,
                                             monitor='val_accuracy',
                                             verbose=1,
                                             save_best_only=True)

    lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)
    progress_bar = keras.callbacks.ProgbarLogger()
    callbacks = [checkpoint, lr_scheduler, progress_bar]

    # Train the model
    train_dataset.shuffle()
    history = model.fit(train_dataset.X, train_dataset.y,
              batch_size=batch_size,
              epochs=num_epochs,
              validation_data=(val_dataset.X, val_dataset.y),
              shuffle=True,
              callbacks=callbacks)

    # Save the final model
    model.save(f'{save_dir}/{model_type}_final_model.h5')

    # Test set evaluation
    print('Loading test data...')
    test_dataset = Dataset(test_file, data_format='channel_last')

    print("Test dataset shape:")
    for k, v in test_dataset.X.items():
        print(f"  {k}: {v.shape}")
    print(f"Test labels shape: {test_dataset.y.shape}")

    # Evaluate on test set
    model_dir = 'roc_info'
    os.makedirs(model_dir, exist_ok=True)

    predictions = model.predict(test_dataset.X)
    auc_score = roc_auc_score(test_dataset.y, predictions[:, 1])  # Assuming binary classification

    # Plot and save ROC curve
    fpr, tpr, roc_auc = plot_roc_curve(test_dataset.y, predictions[:, 1], model_dir)
    save_roc_data(fpr, tpr, roc_auc, model_dir)
    save_auc_score(model_dir, auc_score)

    print(f"AUC Score: {auc_score:.9f}")