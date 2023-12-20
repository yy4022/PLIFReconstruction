import json

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from methods_show import show_loss

"""
PART 1 -- Compare the MSE loss during the training process for all models.
NOTE: compare the scaled MSE loss for the whole training dataset (90% training data + 10% validation data)
-- currently, (75% training data + 25% validation data)
"""

# # STEP 0: define the default parameters
# boxes = 12
#
# # STEP 1: process the MSE loss for the global model
# global_train_loss_records = []
# global_validation_loss_records = []
#
# global_train_loss_records = np.array(global_train_loss_records)
# global_validation_loss_records = np.array(global_validation_loss_records)
#
# # load the MSE loss data
# global_train_loss_records = \
#     np.append(global_train_loss_records, np.load('./result/global/train_loss_records.npy'))
#
# global_validation_loss_records = \
#     np.append(global_validation_loss_records, np.load('./result/global/validation_loss_records.npy'))
#
# # get the mean MSE loss for the whole training dataset
# global_loss_records = (global_train_loss_records * 3) + global_validation_loss_records
# global_loss_records = global_loss_records / 4
#
# # STEP 2: plot the MSE loss for the global model first
# plt.figure(figsize=(10, 8))
#
# plt.semilogy(global_loss_records, label='Global', color='red')
#
# # STEP 3: loop to process the MSE loss for the local models
# for i in range(boxes):
#     local_train_loss_records_i = []
#     local_validation_loss_records_i = []
#
#     local_train_loss_records_i = np.array(local_train_loss_records_i)
#     local_validation_loss_records_i = np.array(local_validation_loss_records_i)
#
#     # load the MSE loss data
#     local_train_loss_records_i = (
#         np.append(local_train_loss_records_i, np.load(f'./result/local/train_loss_records_box{i + 1}.npy')))
#
#     local_validation_loss_records_i = (
#         np.append(local_validation_loss_records_i, np.load(f'./result/local/validation_loss_records_box{i + 1}.npy')))
#
#     # get the mean MSE loss for the whole training dataset
#     local_loss_records_i = (local_train_loss_records_i * 3) + local_validation_loss_records_i
#     local_loss_records_i = local_loss_records_i / 4
#
#     # plot the MSE loss line for box-i
#     plt.semilogy(local_loss_records_i, label='Local', color='black')
#
# # STEP 4: adjust the settings of the figure
# plt.xlim(0, 10000)
# # plt.ylim(0, 1)
#
# plt.xlabel('Epochs')
# plt.ylabel('MSE')
#
# global_line = Line2D([0], [0], color='red', label='Global')
# local_line = Line2D([0], [0], color='black', label='Local')
#
# legend_elements = [global_line] + [local_line]
# legend = plt.legend(handles=legend_elements, loc='upper right')
# for line, label in zip(legend.get_lines(), ['Global'] + ['Local']):
#     line.set_label(label)
#
# # plt.savefig(f"./result/{filename}")
# plt.title('PIV-x')
# plt.show()
#
# """
# PART 2: calculate the unscaled MSE loss of every box for both models
# """
#
# # STEP 1: load the min and max values of all datasets
# with open('data/Preprocessed_Data_old/dataset_information.json', 'r') as file:
#     loaded_data = json.load(file)
#
# print(loaded_data)

train_loss_records = []
validation_loss_records = []

train_loss_records = np.array(train_loss_records)
validation_loss_records = np.array(validation_loss_records)

train_loss_records = \
    np.append(train_loss_records, np.load('./result/train_loss_records.npy'))

validation_loss_records = \
    np.append(validation_loss_records, np.load('./result/validation_loss_records.npy'))

loss_records = {
    'train_loss_records': train_loss_records,
    'validation_loss_records': validation_loss_records
}

# PART 5: show the results
# 5.1. show the loss records of the whole training process
show_loss(loss_records, "fullyCNN_loss.png")
