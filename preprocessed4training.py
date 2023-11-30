import os.path
import pickle

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
files_PIV = [
             # 'data/Attached state/D1F1_air240_PIV_1001to2000.mat',
             # 'data/Attached state/D1F1_air240_PIV_2001to3000.mat',
             # 'data/Detachment process/D1F1_air240_PIV_13401to13600.mat',
             'data/Lifted state/D1F1_air240_PIV_13601to14000.mat',
             'data/Lifted state/D1F1_air240_PIV_14001to14999.mat',
             'data/Reattachment process/D1F1_air240_PIV_451to650.mat',
             'data/Reattachment process/D1F1_air240_PIV_6201to6700.mat'
             ]

files_PLIF = [
              # 'data/Attached state/D1F1_air240_PLIF_1001to2000.mat',
              # 'data/Attached state/D1F1_air240_PLIF_2001to3000.mat',
              # 'data/Detachment process/D1F1_air240_PLIF_13401to13600.mat',
              'data/Lifted state/D1F1_air240_PLIF_13601to14000.mat',
              'data/Lifted state/D1F1_air240_PLIF_14001to14999.mat',
              'data/Reattachment process/D1F1_air240_PLIF_451to650.mat',
              'data/Reattachment process/D1F1_air240_PLIF_6201to6700.mat'
              ]

# PART 2: preprocess the datasets (for PLIF)
# 2.1. preprocess the datasets, then return the cropped datasets
cropped_PLIF_data = crop_old_PLIFdata(files_PLIF)

# 2.2. get the min and max value for all PLIF datasets
min_PLIF, max_PLIF = get_min_max(cropped_PLIF_data)

# 2.3. normalize and discretize the datasets according to the min, max values
preprocessed_PLIF_data = preprocess_data_list(cropped_PLIF_data, min_PLIF, max_PLIF)

# PART 3: split the datasets to training and validation datasets for training(for PLIF)
# 3.1. concatenate the datasets as required
PLIF_data = concatenate_data(preprocessed_PLIF_data)

# 3.2. split the datasets for training, validation and testing
# 3.2.1. get the total number of data
data_num = PLIF_data.shape[1]

# 3.2.2 shuffle the datasets by the index
# create a shuffled index array
index_array = np.arange(data_num)
np.random.shuffle(index_array)

# shuffle the datasets according to the same shuffled index array
PLIF_data = np.take(PLIF_data, index_array, axis=1)

# 3.2.3. calculate the position of splitting points to satisfy the proportion (3: 1: 1)
split_points = [int(np.floor(data_num * 0.6)), int(np.floor(data_num * 0.8))]

# 3.2.4. split the datasets according to the splitting points
PLIF_data_split = np.split(PLIF_data, split_points, axis=1)

# 3.2.5. obtain the training, validation sets
training_PLIF_data = PLIF_data_split[0]
validation_PLIF_data = PLIF_data_split[1]

training_nums = training_PLIF_data.shape[1]
validation_nums = validation_PLIF_data.shape[1]

# 3.3. reshape the training, validation datasets
# get the essential shape information for reshaping datasets
boxes = training_PLIF_data.shape[0] # 12 boxes

PLIF_height = training_PLIF_data.shape[2]
PLIF_width = training_PLIF_data.shape[3]

# then reshape datasets (i.e. flatten the 12 boxes)
training_PLIF_data = training_PLIF_data.reshape((boxes * training_nums, PLIF_height, PLIF_width))
validation_PLIF_data = validation_PLIF_data.reshape((boxes * validation_nums, PLIF_height, PLIF_width))

# 3.4. obtain the training, validation sets
training_PLIF_data = np.expand_dims(training_PLIF_data, axis=1)
validation_PLIF_data = np.expand_dims(validation_PLIF_data, axis=1)

# 3.5. create the corresponding datasets
training_PLIF_dataset = MyDataset(training_PLIF_data)
validation_PLIF_dataset = MyDataset(validation_PLIF_data)

# 3.6. save the datasets for training
with open('data/Preprocessed_Data_old/Train/training_PLIF_dataset1.pkl', 'wb') as file:
    pickle.dump(training_PLIF_dataset, file)

with open('data/Preprocessed_Data_old/Train/validation_PLIF_dataset1.pkl', 'wb') as file:
    pickle.dump(validation_PLIF_dataset, file)

# PART 4: preprocess the datasets (for PIV)
# 4.1. preprocess the datasets, then return the cropped datasets
cropped_PIV_x_data, cropped_PIV_y_data, cropped_PIV_z_data = crop_old_PIVdata(files_PIV)

# 4.2. get the min and max value for all PIV-x, y, z datasets
min_PIV_x, max_PIV_x = get_min_max(cropped_PIV_x_data)

# 4.3. normalize and discretize the datasets according to the min, max values
preprocessed_PIV_x_data = preprocess_data_list(cropped_PIV_x_data, min_PIV_x, max_PIV_x)

# PART 5: split the datasets to training and validation datasets for training (for PIV)
# 5.1. concatenate the datasets as required
PIV_x_data = concatenate_data(preprocessed_PIV_x_data)

# 5.2. split the datasets for training, validation
# 5.2.1. shuffle the datasets according to the same shuffled index array
PIV_x_data = np.take(PIV_x_data, index_array, axis=1)

# 5.2.2. split the datasets according to the splitting points
PIV_x_data_split = np.split(PIV_x_data, split_points, axis=1)

# 5.2.3. obtain the training, validation sets
training_x_PIV_data = PIV_x_data_split[0]
validation_x_PIV_data = PIV_x_data_split[1]

# 5.3. reshape the training, validation datasets
# get the essential shape information for reshaping datasets
PIV_x_height = training_x_PIV_data.shape[2]
PIV_x_width = training_x_PIV_data.shape[3]

# then reshape datasets (i.e. flatten the 12 boxes)
training_x_PIV_data = training_x_PIV_data.reshape((boxes * training_nums, PIV_x_height, PIV_x_width))
validation_x_PIV_data = validation_x_PIV_data.reshape((boxes * validation_nums, PIV_x_height, PIV_x_width))

# 5.4. obtain the training, validation sets
training_x_PIV_data = np.expand_dims(training_x_PIV_data, axis=1)
validation_x_PIV_data = np.expand_dims(validation_x_PIV_data, axis=1)

# 5.5. create the corresponding datasets
training_x_PIV_dataset = MyDataset(training_x_PIV_data)
validation_x_PIV_dataset = MyDataset(validation_x_PIV_data)

# 5.6. save the datasets for training
with open('data/Preprocessed_Data_old/Train/training_x_PIV_dataset1.pkl', 'wb') as file:
    pickle.dump(training_x_PIV_dataset, file)

with open('data/Preprocessed_Data_old/Train/validation_x_PIV_dataset1.pkl', 'wb') as file:
    pickle.dump(validation_x_PIV_dataset, file)
