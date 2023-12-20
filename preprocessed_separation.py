import json
import pickle

import numpy as np

from methods_preprocess import crop_old_PLIFdata, get_min_max, crop_old_PIVdata, MyDataset

"""
This file is used for preprocessing PLIF and PIV datasets simultaneously.
Then, we will obtain the datasets for training and evaluating the model.
(NOTE: loop to preprocess the training and validation datasets, but do not shuffle the testing dataset.)
"""

# SECTION 1: process for the training and validation datasets
# PART 1: provide the essential information
dataset_nums = 7

# PART 2: loop to process the datasets
for i in range(dataset_nums):
    # Step 1: load the datasets
    PLIF_data = np.load(f'data/PLIF240air/PLIF_dataset{i + 1}.npy')
    PIV_x_data = np.load(f'data/Full dataset/PIV_x_dataset{i + 1}.npy')
    PIV_y_data = np.load(f'data/Full dataset/PIV_y_dataset{i + 1}.npy')
    PIV_z_data = np.load(f'data/Full dataset/PIV_z_dataset{i + 1}.npy')

    # Step 2: get the total number of data
    data_num = PLIF_data.shape[1]

    # Step 3: shuffle the datasets by the index
    # create a shuffled index array
    index_array = np.arange(data_num)
    np.random.shuffle(index_array)

    # shuffle the datasets according to the same shuffled index array
    PLIF_data = np.take(PLIF_data, index_array, axis=1)
    PIV_x_data = np.take(PIV_x_data, index_array, axis=1)
    PIV_y_data = np.take(PIV_y_data, index_array, axis=1)
    PIV_z_data = np.take(PIV_z_data, index_array, axis=1)

    # Step 4: split the datasets
    split_points = [int(np.floor(data_num * 0.9))]

    # split the datasets according to the splitting points
    PLIF_data_split = np.split(PLIF_data, split_points, axis=1)
    PIV_x_data_split = np.split(PIV_x_data, split_points, axis=1)
    PIV_y_data_split = np.split(PIV_y_data, split_points, axis=1)
    PIV_z_data_split = np.split(PIV_z_data, split_points, axis=1)

    # obtain the training and validation sets
    training_PLIF_data = PLIF_data_split[0]
    validation_PLIF_data = PLIF_data_split[1]
    training_PIV_x_data = PIV_x_data_split[0]
    validation_PIV_x_data = PIV_x_data_split[1]
    training_PIV_y_data = PIV_y_data_split[0]
    validation_PIV_y_data = PIV_y_data_split[1]
    training_PIV_z_data = PIV_z_data_split[0]
    validation_PIV_z_data = PIV_z_data_split[1]

    training_nums = training_PLIF_data.shape[1]
    validation_nums = validation_PLIF_data.shape[1]

    # save the training and validation data (shuffled numpy array data)
    np.save(f'data/Preprocessed_Data_Fulldataset/data_PLIF/training_PLIF_data{i + 1}.npy', training_PLIF_data)
    np.save(f'data/Preprocessed_Data_Fulldataset/data_PLIF/validation_PLIF_data{i + 1}.npy', validation_PLIF_data)
    np.save(f'data/Preprocessed_Data_Fulldataset/data_PIV/training_PIV_x_data{i + 1}.npy', training_PIV_x_data)
    np.save(f'data/Preprocessed_Data_Fulldataset/data_PIV/validation_PIV_x_data{i}.npy', validation_PIV_x_data)
    np.save(f'data/Preprocessed_Data_Fulldataset/data_PIV/training_PIV_y_data{i + 1}.npy', training_PIV_y_data)
    np.save(f'data/Preprocessed_Data_Fulldataset/data_PIV/validation_PIV_y_data{i}.npy', validation_PIV_y_data)
    np.save(f'data/Preprocessed_Data_Fulldataset/data_PIV/training_PIV_z_data{i + 1}.npy', training_PIV_z_data)
    np.save(f'data/Preprocessed_Data_Fulldataset/data_PIV/validation_PIV_z_data{i}.npy', validation_PIV_z_data)

    # Step 5: reshape the datasets
    # get the essential shape information for reshaping datasets
    boxes = training_PLIF_data.shape[0]  # 12 boxes

    PLIF_height = training_PLIF_data.shape[2]
    PLIF_width = training_PLIF_data.shape[3]
    PIV_x_height = training_PIV_x_data.shape[2]
    PIV_x_width = training_PIV_x_data.shape[3]
    PIV_y_height = training_PIV_y_data.shape[2]
    PIV_y_width = training_PIV_y_data.shape[3]
    PIV_z_height = training_PIV_z_data.shape[2]
    PIV_z_width = training_PIV_z_data.shape[3]

    # then reshape datasets (i.e. flatten the 12 boxes)
    training_PLIF_data = training_PLIF_data.reshape((boxes * training_nums, PLIF_height, PLIF_width))
    validation_PLIF_data = validation_PLIF_data.reshape((boxes * validation_nums, PLIF_height, PLIF_width))
    training_PIV_x_data = training_PIV_x_data.reshape((boxes * training_nums, PIV_x_height, PIV_x_width))
    validation_PIV_x_data = validation_PIV_x_data.reshape((boxes * validation_nums, PIV_x_height, PIV_x_width))
    training_PIV_y_data = training_PIV_y_data.reshape((boxes * training_nums, PIV_y_height, PIV_y_width))
    validation_PIV_y_data = validation_PIV_y_data.reshape((boxes * validation_nums, PIV_y_height, PIV_y_width))
    training_PIV_z_data = training_PIV_z_data.reshape((boxes * training_nums, PIV_z_height, PIV_z_width))
    validation_PIV_z_data = validation_PIV_z_data.reshape((boxes * validation_nums, PIV_z_height, PIV_z_width))

    # obtain the training, validation and testing sets
    training_PLIF_data = np.expand_dims(training_PLIF_data, axis=1)
    validation_PLIF_data = np.expand_dims(validation_PLIF_data, axis=1)
    training_PIV_x_data = np.expand_dims(training_PIV_x_data, axis=1)
    validation_PIV_x_data = np.expand_dims(validation_PIV_x_data, axis=1)
    training_PIV_y_data = np.expand_dims(training_PIV_y_data, axis=1)
    validation_PIV_y_data = np.expand_dims(validation_PIV_y_data, axis=1)
    training_PIV_z_data = np.expand_dims(training_PIV_z_data, axis=1)
    validation_PIV_z_data = np.expand_dims(validation_PIV_z_data, axis=1)

    # create the corresponding datasets
    training_PLIF_dataset = MyDataset(training_PLIF_data)
    validation_PLIF_dataset = MyDataset(validation_PLIF_data)
    training_PIV_x_dataset = MyDataset(training_PIV_x_data)
    validation_PIV_x_dataset = MyDataset(validation_PIV_x_data)
    training_PIV_y_dataset = MyDataset(training_PIV_y_data)
    validation_PIV_y_dataset = MyDataset(validation_PIV_y_data)
    training_PIV_z_dataset = MyDataset(training_PIV_z_data)
    validation_PIV_z_dataset = MyDataset(validation_PIV_z_data)

    # Step 6. save the datasets for training
    with open(f'data/Preprocessed_Data_Fulldataset/data_PLIF/training_PLIF_dataset{i + 1}.pkl', 'wb') as file:
        pickle.dump(training_PLIF_dataset, file)

    with open(f'data/Preprocessed_Data_Fulldataset/data_PLIF/validation_PLIF_dataset{i + 1}.pkl', 'wb') as file:
        pickle.dump(validation_PLIF_dataset, file)

    with open(f'data/Preprocessed_Data_Fulldataset/data_PIV/training_PIV_x_dataset{i + 1}.pkl', 'wb') as file:
        pickle.dump(training_PIV_x_dataset, file)

    with open(f'data/Preprocessed_Data_Fulldataset/data_PIV/validation_PIV_x_dataset{i + 1}.pkl', 'wb') as file:
        pickle.dump(validation_PIV_x_dataset, file)

    with open(f'data/Preprocessed_Data_Fulldataset/data_PIV/training_PIV_y_dataset{i + 1}.pkl', 'wb') as file:
        pickle.dump(training_PIV_y_dataset, file)

    with open(f'data/Preprocessed_Data_Fulldataset/data_PIV/validation_PIV_y_dataset{i + 1}.pkl', 'wb') as file:
        pickle.dump(validation_PIV_y_dataset, file)

    with open(f'data/Preprocessed_Data_Fulldataset/data_PIV/training_PIV_z_dataset{i + 1}.pkl', 'wb') as file:
        pickle.dump(training_PIV_z_dataset, file)

    with open(f'data/Preprocessed_Data_Fulldataset/data_PIV/validation_PIV_z_dataset{i + 1}.pkl', 'wb') as file:
        pickle.dump(validation_PIV_z_dataset, file)

# SECTION 2: prepare for the testing dataset
# PART 1: provide the essential information
specified_num = 8

# PART 2: process the testing dataset
# Step 1: load the datasets
PLIF_data = np.load(f'data/PLIF240air/PLIF_dataset{specified_num}.npy')
PIV_x_data = np.load(f'data/Full dataset/PIV_x_dataset{specified_num}.npy')
PIV_y_data = np.load(f'data/Full dataset/PIV_y_dataset{specified_num}.npy')
PIV_z_data = np.load(f'data/Full dataset/PIV_z_dataset{specified_num}.npy')

# Step 2: save the testing data, directly (not shuffled)
np.save(f'data/Preprocessed_Data_Fulldataset/data_PLIF/testing_PLIF_data{specified_num}.npy', PLIF_data)
np.save(f'data/Preprocessed_Data_Fulldataset/data_PIV/testing_PIV_x_data{specified_num}.npy', PIV_x_data)
np.save(f'data/Preprocessed_Data_Fulldataset/data_PIV/testing_PIV_y_data{specified_num}.npy', PIV_y_data)
np.save(f'data/Preprocessed_Data_Fulldataset/data_PIV/testing_PIV_z_data{specified_num}.npy', PIV_z_data)

# Step 3: reshape the datasets
# get the essential shape information for reshaping datasets
testing_nums = PLIF_data.shape[1]
boxes = PLIF_data.shape[0]  # 12 boxes

PLIF_height = PLIF_data.shape[2]
PLIF_width = PLIF_data.shape[3]
PIV_x_height = PIV_x_data.shape[2]
PIV_x_width = PIV_x_data.shape[3]
PIV_y_height = PIV_y_data.shape[2]
PIV_y_width = PIV_y_data.shape[3]
PIV_z_height = PIV_z_data.shape[2]
PIV_z_width = PIV_z_data.shape[3]

# then reshape datasets (i.e. flatten the 12 boxes)
testing_PLIF_data = PLIF_data.reshape((boxes * testing_nums, PLIF_height, PLIF_width))
testing_PIV_x_data = PIV_x_data.reshape((boxes * testing_nums, PIV_x_height, PIV_x_width))
testing_PIV_y_data = PIV_y_data.reshape((boxes * testing_nums, PIV_y_height, PIV_y_width))
testing_PIV_z_data = PIV_z_data.reshape((boxes * testing_nums, PIV_z_height, PIV_z_width))

# obtain the training, validation and testing sets
testing_PLIF_data = np.expand_dims(testing_PLIF_data, axis=1)
testing_PIV_x_data = np.expand_dims(testing_PIV_x_data, axis=1)
testing_PIV_y_data = np.expand_dims(testing_PIV_y_data, axis=1)
testing_PIV_z_data = np.expand_dims(testing_PIV_z_data, axis=1)

# create the corresponding datasets
testing_PLIF_dataset = MyDataset(testing_PLIF_data)
testing_PIV_x_dataset = MyDataset(testing_PIV_x_data)
testing_PIV_y_dataset = MyDataset(testing_PIV_y_data)
testing_PIV_z_dataset = MyDataset(testing_PIV_z_data)

# Step 6. save the datasets for training
with open(f'data/Preprocessed_Data_Fulldataset/data_PLIF/testing_PLIF_dataset{specified_num}.pkl', 'wb') as file:
    pickle.dump(testing_PLIF_dataset, file)

with open(f'data/Preprocessed_Data_Fulldataset/data_PIV/testing_PIV_x_dataset{specified_num}.pkl', 'wb') as file:
    pickle.dump(testing_PIV_x_dataset, file)

with open(f'data/Preprocessed_Data_Fulldataset/data_PIV/testing_PIV_y_dataset{specified_num}.pkl', 'wb') as file:
    pickle.dump(testing_PIV_y_dataset, file)

with open(f'data/Preprocessed_Data_Fulldataset/data_PIV/testing_PIV_z_dataset{specified_num}.pkl', 'wb') as file:
    pickle.dump(testing_PIV_z_dataset, file)
