import os.path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import summary

from fullyCNN.train import train_epoch
from fullyCNN.validate import validate_epoch
from preprocess_methods import MyDataset, crop_old_PIVdata, crop_old_PLIFdata, \
    get_min_max, preprocess_data_list, concatenate_data
from fullyCNN.neural_net import FullyCNN
from result_visualiser import show_loss

"""
This file is used for testing the whole process of training the global model.
NOTE: there is no need to use the testing datasets during the training process.
"""

# PART 1: define the parameters
# 1.1. define the parameters for training the model
batch_size = 100
rows = 3
columns = 4

# 1.2. provide filenames of PIV, PLIF data
files_PIV = ['data/Attached state/D1F1_air240_PIV_1001to2000.mat',
             'data/Attached state/D1F1_air240_PIV_2001to3000.mat',
             'data/Detachment process/D1F1_air240_PIV_13401to13600.mat',
             'data/Lifted state/D1F1_air240_PIV_13601to14000.mat',
             'data/Lifted state/D1F1_air240_PIV_14001to14999.mat',
             'data/Reattachment process/D1F1_air240_PIV_451to650.mat',
             'data/Reattachment process/D1F1_air240_PIV_6201to6700.mat']

files_PLIF = ['data/Attached state/D1F1_air240_PLIF_1001to2000.mat',
              'data/Attached state/D1F1_air240_PLIF_2001to3000.mat',
              'data/Detachment process/D1F1_air240_PLIF_13401to13600.mat',
              'data/Lifted state/D1F1_air240_PLIF_13601to14000.mat',
              'data/Lifted state/D1F1_air240_PLIF_14001to14999.mat',
              'data/Reattachment process/D1F1_air240_PLIF_451to650.mat',
              'data/Reattachment process/D1F1_air240_PLIF_6201to6700.mat']

# PART 2: preprocess the datasets (for PLIF)
# 2.1. preprocess the datasets, then return the cropped datasets
cropped_PLIF_data = crop_old_PLIFdata(files_PLIF)

# 2.2. get the min and max value for all PLIF datasets
min_PLIF, max_PLIF = get_min_max(cropped_PLIF_data)

# 2.3. normalize and discretize the datasets according to the min, max values
preprocessed_PLIF_data = preprocess_data_list(cropped_PLIF_data, min_PLIF, max_PLIF)

# PART 3: split the datasets to training, validation, testing datasets (for PLIF)
# 1. concatenate the datasets as required
PLIF_data = concatenate_data(preprocessed_PLIF_data)

# 2. split the datasets for training, validation and testing
# 2.1. get the total number of data
data_num = PLIF_data.shape[1]

# 2.2 shuffle the datasets by the index
# create a shuffled index array
index_array = np.arange(data_num)
np.random.shuffle(index_array)

# shuffle the datasets according to the same shuffled index array
PLIF_data = np.take(PLIF_data, index_array, axis=1)

# 2.2. calculate the position of splitting points to satisfy the proportion (3: 1: 1)
split_points = [int(np.floor(data_num * 0.6)), int(np.floor(data_num * 0.8))]

# 2.3. split the datasets according to the splitting points
PLIF_data_split = np.split(PLIF_data, split_points, axis=1)

# 2.4. obtain the training, validation, testing sets
training_PLIF_data = PLIF_data_split[0]
validation_PLIF_data = PLIF_data_split[1]
testing_PLIF_data = PLIF_data_split[2]

# 2.5. save the datasets
np.save('data/Preprocessed_Data_old/training_PLIF_data.npy', training_PLIF_data)
np.save('data/Preprocessed_Data_old/validation_PLIF_data.npy', validation_PLIF_data)
np.save('data/Preprocessed_Data_old/testing_PLIF_data.npy', testing_PLIF_data)

# PART 4: preprocess the datasets (for PIV)
# 4.1. preprocess the datasets, then return the cropped datasets
cropped_PIV_x_data, cropped_PIV_y_data, cropped_PIV_z_data = crop_old_PIVdata(files_PIV)

# 4.2. get the min and max value for all PIV-x, y, z datasets
min_PIV_x, max_PIV_x = get_min_max(cropped_PIV_x_data)
# min_PIV_y, max_PIV_y = get_min_max(cropped_PIV_y_data)
# min_PIV_z, max_PIV_z = get_min_max(cropped_PIV_z_data)

# 4.3. normalize and discretize the datasets according to the min, max values
preprocessed_PIV_x_data = preprocess_data_list(cropped_PIV_x_data, min_PIV_x, max_PIV_x)
# preprocessed_PIV_y_data = preprocess_data_list(cropped_PIV_y_data, min_PIV_y, max_PIV_y)
# preprocessed_PIV_z_data = preprocess_data_list(cropped_PIV_z_data, min_PIV_z, max_PIV_z)

# PART 5: split the datasets to training, validation, testing datasets
# 1. concatenate the datasets as required
PIV_x_data = concatenate_data(preprocessed_PIV_x_data)
# PIV_y_data = concatenate_data(preprocessed_PIV_y_data)
# PIV_z_data = concatenate_data(preprocessed_PIV_z_data)

# 2. split the datasets for training, validation and testing
# 2.1. shuffle the datasets according to the same shuffled index array
PIV_x_data = np.take(PIV_x_data, index_array, axis=1)
# PIV_y_data = np.take(PIV_y_data, index_array, axis=1)
# PIV_z_data = np.take(PIV_z_data, index_array, axis=1)

# 2.2. split the datasets according to the splitting points
PIV_x_data_split = np.split(PIV_x_data, split_points, axis=1)

# 2.4. obtain the training, validation, testing sets
training_x_PIV_data = PIV_x_data_split[0]
validation_x_PIV_data = PIV_x_data_split[1]
testing_x_PIV_data = PIV_x_data_split[2]

# 2.5. save the datasets
np.save('data/Preprocessed_Data_old/training_x_PIV_data.npy', training_x_PIV_data)
np.save('data/Preprocessed_Data_old/validation_x_PIV_data.npy', validation_x_PIV_data)
np.save('data/Preprocessed_Data_old/testing_PIV_data.npy', testing_x_PIV_data)
