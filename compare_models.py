import numpy as np
from matplotlib import pyplot as plt

"""
PART 1 -- Compare the MSE loss during the training process for all models.
NOTE: compare the scaled MSE loss for the whole training dataset (90% training data + 10% validation data)
-- currently, (75% training data + 25% validation data)
"""

# STEP 0: define the default parameters
boxes = 12

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

plt.semilogy(global_loss_records, label='Global', color='red')

# STEP 3: loop to process the MSE loss for the local models
for i in range(boxes):
    local_train_loss_records_i = []
    local_validation_loss_records_i = []

    local_train_loss_records_i = np.array(local_train_loss_records_i)
    local_validation_loss_records_i = np.array(local_validation_loss_records_i)

    # load the MSE loss data
    local_train_loss_records_i = (
        np.append(local_train_loss_records_i, np.load(f'./result/local/train_loss_records_box{i + 1}.npy')))

    local_validation_loss_records_i = (
        np.append(local_validation_loss_records_i, np.load(f'./result/local/validation_loss_records_box{i + 1}.npy')))

    # get the mean MSE loss for the whole training dataset
    local_loss_records_i = (local_train_loss_records_i * 3) + local_validation_loss_records_i
    local_loss_records_i = local_loss_records_i / 4

    # plot the MSE loss line for box-i
    plt.semilogy(local_loss_records_i, label=f'Local-box{i + 1}', color='black')

#
plt.xlim(0, 10000)
# plt.ylim(1e-2, 1.2e-2)

plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()

# plt.savefig(f"./result/{filename}")
plt.show()
