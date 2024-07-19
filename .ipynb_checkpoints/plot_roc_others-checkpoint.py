import os
import numpy as np
import matplotlib.pyplot as plt

# Directory containing the ROC data files
roc_data_dir = 'roc_info'  # Update this path

# List of NPZ files and their corresponding model names
roc_files = [
    ('dsa_roc_data_test.npz', 'DSA'),
    ('dsapre_roc_data_test.npz', 'DSA Pre'),
    ('efn_roc_data_test.npz', 'EFN'),
    ('efnpre_roc_data_test.npz', 'EFN Pre'),
    ('pfn_roc_data_test.npz', 'PFN'),
    ('pfnpre_roc_data_test.npz', 'PFN Pre'),
]

# Plot settings
plt.figure(figsize=(10, 8))
lw = 2

# Plot ROC curves for each model
for roc_file, model_name in roc_files:
    # Load the ROC data
    data = np.load(os.path.join(roc_data_dir, roc_file))
    fpr = data['fpr']
    tpr = data['tpr']
    roc_auc = data['roc_auc']

    # Plot the ROC curve
    plt.plot(tpr, 1/fpr, lw=lw, linestyle='--', label=f'{model_name} (AUC = {roc_auc:.5f})')

# Plot the reference line
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

# Plot settings
#plt.xscale('log')
plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
plt.yscale('log')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison of Different Models')
plt.legend(loc="lower right")

# Save the combined ROC plot
plt.savefig(os.path.join(roc_data_dir, 'combined_roc_curve.png'))

# Show the plot
plt.show()
