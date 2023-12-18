import numpy as np
from matplotlib import pyplot as plt

"""
NOTE: compare the scaled MSE loss for the whole training dataset (90% training data + 10% validation data)
-- currently, (75% training data + 25% validation data)
"""

# STEP 1: process the MSE loss for the global model
global_train_loss_records = []
global_validation_loss_records = []

global_train_loss_records = np.array(global_train_loss_records)
global_validation_loss_records = np.array(global_validation_loss_records)

# load the MSE loss data
global_train_loss_records = \
    np.append(global_train_loss_records, np.load('./result/global/train_loss_records.npy'))

global_validation_loss_records = \
    np.append(global_validation_loss_records, np.load('./result/global/validation_loss_records.npy'))

# get the mean MSE loss for the whole training dataset
global_loss_records = (global_train_loss_records * 3) + global_validation_loss_records
global_loss_records = global_loss_records / 4

# STEP 2: plot the MSE loss for the global model first
plt.figure(figsize=(10, 8))

plt.semilogy(global_train_loss_records, label='Global_train')
plt.semilogy(global_validation_loss_records, label='Global_validation')
plt.semilogy(global_loss_records, label='Global')

plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()

# plt.savefig(f"./result/{filename}")
plt.show()
