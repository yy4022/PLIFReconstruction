import json

from methods_preprocess import crop_old_PLIFdata, get_min_max, crop_old_PIVdata

"""
SECTION 1: This section is used for getting the global min and max values of PLIF and PIV dataset for normalization.
(NOTE:  1. make sure that there is no dataset_information.json file in the directory
        2. you can comment this section after implementing it)
"""

# SECTION 1: get the global min and max value of PLIF and PIV dataset.
# PART 1: process the PLIF datasets
# Step 1: define the parameters for training the model
specified_dataset = 2

# provide the filename of PLIF data
files_PLIF = [
    # 'data/PLIF240air/D1F1_air240_PLIF_1to2000.mat',  # training + validation - dataset1 (2000)
    # 'data/PLIF240air/D1F1_air240_PLIF_2001to4000.mat', # training + validation - dataset2 (2000)
    # 'data/PLIF240air/D1F1_air240_PLIF_4001to6000.mat',  # training + validation - dataset3 (2000)
    # 'data/PLIF240air/D1F1_air240_PLIF_6001to8000.mat',  # training + validation - dataset4 (2000)
    'data/PLIF240air/D1F1_air240_PLIF_8001to10000.mat',  # training + validation - dataset5 (2000)
    'data/PLIF240air/D1F1_air240_PLIF_10001to12000.mat',  # training + validation - dataset6 (2000)
    'data/PLIF240air/D1F1_air240_PLIF_12001to14000.mat',  # training + validation - dataset7 (2000)
    # 'data/PLIF240air/D1F1_air240_PLIF_14001to14999.mat',  # testing - dataset8 (999)
]

# Step 2: get the min and max value for this specified dataset (for PLIF)
# 2.1. preprocess the datasets, then return the cropped datasets
cropped_PLIF_data = crop_old_PLIFdata(files_PLIF)

# 2.2. get the min and max value for this PLIF dataset
min_PLIF, max_PLIF = get_min_max(cropped_PLIF_data)

# Step 3: compare the min and max value with the saved one
# 3.1. try to load the existing file
try:
    # if the values have existed, compare and update the value
    with open('data/Preprocessed_Data_Fulldataset/dataset_information_PLIF.json', 'r') as file:
        existing_data = json.load(file)

    current_min_PLIF = existing_data['min_PLIF']
    current_max_PLIF = existing_data['max_PLIF']

    if current_min_PLIF > min_PLIF:
        existing_data['min_PLIF'] = min_PLIF
    if current_max_PLIF < max_PLIF:
        existing_data['max_PLIF'] = max_PLIF

except FileNotFoundError:
    # if the values have not existed, create a new one
    existing_data = {}

    # add new information to the file
    new_data = {
        'min_PLIF': float(min_PLIF),
        'max_PLIF': float(max_PLIF),
    }

    existing_data.update(new_data)

print(existing_data)

# 3.2. save the updated data information
with open('data/Preprocessed_Data_Fulldataset/dataset_information_PLIF.json', 'w') as file:
    json.dump(existing_data, file)

# PART 2: preprocess the PIV dataset
# Step 1: provide the filename of PIV data
files_PIV = [
    'data/Full dataset/D1F1_air240_PIV_1to14999.mat',
]

# Step 2: get the min and max value for this specified dataset (for PIV)
# 2.1. preprocess the datasets, then return the cropped datasets
cropped_PIV_x_data, cropped_PIV_y_data, cropped_PIV_z_data = crop_old_PIVdata(files_PIV)

# 2.2. get the min and max value for this PIV-x, y, z dataset
min_PIV_x, max_PIV_x = get_min_max(cropped_PIV_x_data)
min_PIV_y, max_PIV_y = get_min_max(cropped_PIV_y_data)
min_PIV_z, max_PIV_z = get_min_max(cropped_PIV_z_data)

# Step 3: compare the min and max value with the saved one
existing_data = {}

# add new information to the file
new_data = {
    'min_PIV_x': float(min_PIV_x),
    'max_PIV_x': float(max_PIV_x),
    'min_PIV_y': float(min_PIV_y),
    'max_PIV_y': float(max_PIV_y),
    'min_PIV_z': float(min_PIV_z),
    'max_PIV_z': float(max_PIV_z),
}

existing_data.update(new_data)

print(existing_data)

# 3.2. save the updated data information
with open('data/Preprocessed_Data_Fulldataset/dataset_information_PIV.json', 'w') as file:
    json.dump(existing_data, file)

# # 2.3. normalize and discretize the datasets according to the min, max values
# preprocessed_PLIF_data = preprocess_data_list(cropped_PLIF_data, min_PLIF, max_PLIF)

# # PART 3: split the datasets to training, validation and testing datasets for training(for PLIF)
# # 3.1. concatenate the datasets as required
# PLIF_data = concatenate_data(preprocessed_PLIF_data)
#
# # 3.2. split the datasets for training, validation and testing
# # 3.2.1. get the total number of data
# data_num = PLIF_data.shape[1]
#
# # 3.2.2 shuffle the datasets by the index
# # create a shuffled index array
# index_array = np.arange(data_num)
# np.random.shuffle(index_array)
#
# # shuffle the datasets according to the same shuffled index array
# PLIF_data = np.take(PLIF_data, index_array, axis=1)
#
# # 3.2.3. calculate the position of splitting points to satisfy the proportion (3: 1: 1)
# split_points = [int(np.floor(data_num * 0.6)), int(np.floor(data_num * 0.8))]
#
# # 3.2.4. split the datasets according to the splitting points
# PLIF_data_split = np.split(PLIF_data, split_points, axis=1)
#
# # 3.2.5. obtain the training, validation and testing sets
# training_PLIF_data = PLIF_data_split[0]
# validation_PLIF_data = PLIF_data_split[1]
# testing_PLIF_data = PLIF_data_split[2]
#
# training_nums = training_PLIF_data.shape[1]
# validation_nums = validation_PLIF_data.shape[1]
# testing_nums = testing_PLIF_data.shape[1]
#
# # 3.2.6. save the training, validation and testing sets
# np.save(f'data/Preprocessed_Data_old/training_PLIF_data{specified_dataset}.npy', training_PLIF_data)
# np.save(f'data/Preprocessed_Data_old/validation_PLIF_data{specified_dataset}.npy', validation_PLIF_data)
# np.save(f'data/Preprocessed_Data_old/testing_PLIF_data{specified_dataset}.npy', testing_PLIF_data)
#
# # 3.3. reshape the training, validation and testing datasets
# # get the essential shape information for reshaping datasets
# boxes = training_PLIF_data.shape[0]  # 12 boxes
#
# PLIF_height = training_PLIF_data.shape[2]
# PLIF_width = training_PLIF_data.shape[3]
#
# # then reshape datasets (i.e. flatten the 12 boxes)
# training_PLIF_data = training_PLIF_data.reshape((boxes * training_nums, PLIF_height, PLIF_width))
# validation_PLIF_data = validation_PLIF_data.reshape((boxes * validation_nums, PLIF_height, PLIF_width))
# testing_PLIF_data = testing_PLIF_data.reshape((boxes * testing_nums, PLIF_height, PLIF_width))
#
# # 3.4. obtain the training, validation and testing sets
# training_PLIF_data = np.expand_dims(training_PLIF_data, axis=1)
# validation_PLIF_data = np.expand_dims(validation_PLIF_data, axis=1)
# testing_PLIF_data = np.expand_dims(testing_PLIF_data, axis=1)
# s
# # 3.5. create the corresponding datasets
# training_PLIF_dataset = MyDataset(training_PLIF_data)
# validation_PLIF_dataset = MyDataset(validation_PLIF_data)
# testing_PLIF_dataset = MyDataset(testing_PLIF_data)
#
# # 3.6. save the datasets for training
# with open(f'data/Preprocessed_Data_old/training_PLIF_dataset{specified_dataset}.pkl', 'wb') as file:
#     pickle.dump(training_PLIF_dataset, file)
#
# with open(f'data/Preprocessed_Data_old/validation_PLIF_dataset{specified_dataset}.pkl', 'wb') as file:
#     pickle.dump(validation_PLIF_dataset, file)
#
# with open(f'data/Preprocessed_Data_old/testing_PLIF_dataset{specified_dataset}.pkl', 'wb') as file:
#     pickle.dump(testing_PLIF_dataset, file)

# SECTION 2: preprocess the PIV dataset
# provide the filename of PIV data
# files_PIV = [
#     'data/Full dataset/D1F1_air240_PIV_1to14999.mat',
# ]
#
# # PART 4: preprocess the datasets (for PIV)
# # 4.1. preprocess the datasets, then return the cropped datasets
# cropped_PIV_x_data, cropped_PIV_y_data, cropped_PIV_z_data = crop_old_PIVdata(files_PIV)
#
# print(np.shape(cropped_PIV_x_data))

# # 4.2. get the min and max value for all PIV-x, y, z datasets
# min_PIV_x, max_PIV_x = get_min_max(cropped_PIV_x_data)
# min_PIV_y, max_PIV_y = get_min_max(cropped_PIV_y_data)
# min_PIV_z, max_PIV_z = get_min_max(cropped_PIV_z_data)
#
# # 4.3. normalize and discretize the datasets according to the min, max values
# preprocessed_PIV_x_data = preprocess_data_list(cropped_PIV_x_data, min_PIV_x, max_PIV_x)
# preprocessed_PIV_y_data = preprocess_data_list(cropped_PIV_y_data, min_PIV_y, max_PIV_y)
# preprocessed_PIV_z_data = preprocess_data_list(cropped_PIV_z_data, min_PIV_z, max_PIV_z)

# # PART 5: split the datasets to training, validation and testing datasets for training (for PIV)
# # 5.1. concatenate the datasets as required
# PIV_x_data = concatenate_data(preprocessed_PIV_x_data)
#
# # 5.2. split the datasets for training, validation and testing
# # 5.2.1. shuffle the datasets according to the same shuffled index array
# PIV_x_data = np.take(PIV_x_data, index_array, axis=1)
#
# # 5.2.2. split the datasets according to the splitting points
# PIV_x_data_split = np.split(PIV_x_data, split_points, axis=1)
#
# # 5.2.3. obtain the training, validation and testing sets
# training_x_PIV_data = PIV_x_data_split[0]
# validation_x_PIV_data = PIV_x_data_split[1]
# testing_x_PIV_data = PIV_x_data_split[2]
#
# # 5.2.4. save the training, validation and testing sets
# np.save(f'data/Preprocessed_Data_old/training_x_PIV_data{specified_dataset}.npy', training_x_PIV_data)
# np.save(f'data/Preprocessed_Data_old/validation_x_PIV_data{specified_dataset}.npy', validation_x_PIV_data)
# np.save(f'data/Preprocessed_Data_old/testing_x_PIV_data{specified_dataset}.npy', testing_x_PIV_data)
#
# # 5.3. reshape the training, validation and testing datasets
# # get the essential shape information for reshaping datasets
# PIV_x_height = training_x_PIV_data.shape[2]
# PIV_x_width = training_x_PIV_data.shape[3]
#
# # then reshape datasets (i.e. flatten the 12 boxes)
# training_x_PIV_data = training_x_PIV_data.reshape((boxes * training_nums, PIV_x_height, PIV_x_width))
# validation_x_PIV_data = validation_x_PIV_data.reshape((boxes * validation_nums, PIV_x_height, PIV_x_width))
# testing_x_PIV_data = testing_x_PIV_data.reshape((boxes * testing_nums, PIV_x_height, PIV_x_width))
#
# # 5.4. obtain the training, validation and testing sets
# training_x_PIV_data = np.expand_dims(training_x_PIV_data, axis=1)
# validation_x_PIV_data = np.expand_dims(validation_x_PIV_data, axis=1)
# testing_x_PIV_data = np.expand_dims(testing_x_PIV_data, axis=1)
#
# # 5.5. create the corresponding datasets
# training_x_PIV_dataset = MyDataset(training_x_PIV_data)
# validation_x_PIV_dataset = MyDataset(validation_x_PIV_data)
# testing_x_PIV_dataset = MyDataset(testing_x_PIV_data)
#
# # 5.6. save the datasets for training
# with open(f'data/Preprocessed_Data_old/training_x_PIV_dataset{specified_dataset}.pkl', 'wb') as file:
#     pickle.dump(training_x_PIV_dataset, file)
#
# with open(f'data/Preprocessed_Data_old/validation_x_PIV_dataset{specified_dataset}.pkl', 'wb') as file:
#     pickle.dump(validation_x_PIV_dataset, file)
#
# with open(f'data/Preprocessed_Data_old/testing_x_PIV_dataset{specified_dataset}.pkl', 'wb') as file:
#     pickle.dump(testing_x_PIV_dataset, file)

# # PART 1: define the parameters
# # 1.1. define the parameters for training the model
# rows = 3
# columns = 4
# specified_dataset = 4
#
# # 1.2. provide filenames of PIV, PLIF data
# files_PIV = [
#     # 'data/Attached state/D1F1_air240_PIV_1001to2000.mat',  # dataset1 - attached
#     # 'data/Attached state/D1F1_air240_PIV_2001to3000.mat',
#     # 'data/Detachment process/D1F1_air240_PIV_13401to13600.mat',  # dataset2 - detachment
#     # 'data/Lifted state/D1F1_air240_PIV_13601to14000.mat',  # dataset3 - lifted
#     # 'data/Lifted state/D1F1_air240_PIV_14001to14999.mat',
#     'data/Reattachment process/D1F1_air240_PIV_451to650.mat',  # dataset4 - reattachment
#     'data/Reattachment process/D1F1_air240_PIV_6201to6700.mat',
# ]
#
# files_PLIF = [
#     # 'data/Attached state/D1F1_air240_PLIF_1001to2000.mat',  # dataset1 - attached
#     # 'data/Attached state/D1F1_air240_PLIF_2001to3000.mat',
#     # 'data/Detachment process/D1F1_air240_PLIF_13401to13600.mat',  # dataset2 - detachment
#     # 'data/Lifted state/D1F1_air240_PLIF_13601to14000.mat',  # dataset3 - lifted
#     # 'data/Lifted state/D1F1_air240_PLIF_14001to14999.mat',
#     'data/Reattachment process/D1F1_air240_PLIF_451to650.mat',  # dataset4 - reattachment
#     'data/Reattachment process/D1F1_air240_PLIF_6201to6700.mat',
# ]
#
# # PART 2: preprocess the datasets (for PLIF)
# # 2.1. preprocess the datasets, then return the cropped datasets
# cropped_PLIF_data = crop_old_PLIFdata(files_PLIF)
#
# # 2.2. get the min and max value for all PLIF datasets
# min_PLIF, max_PLIF = get_min_max(cropped_PLIF_data)

# # 2.3. normalize and discretize the datasets according to the min, max values
# preprocessed_PLIF_data = preprocess_data_list(cropped_PLIF_data, min_PLIF, max_PLIF)
#
# # PART 3: split the datasets to training, validation and testing datasets for training(for PLIF)
# # 3.1. concatenate the datasets as required
# PLIF_data = concatenate_data(preprocessed_PLIF_data)
#
# # 3.2. split the datasets for training, validation and testing
# # 3.2.1. get the total number of data
# data_num = PLIF_data.shape[1]
#
# # 3.2.2 shuffle the datasets by the index
# # create a shuffled index array
# index_array = np.arange(data_num)
# np.random.shuffle(index_array)
#
# # shuffle the datasets according to the same shuffled index array
# PLIF_data = np.take(PLIF_data, index_array, axis=1)
#
# # 3.2.3. calculate the position of splitting points to satisfy the proportion (3: 1: 1)
# split_points = [int(np.floor(data_num * 0.6)), int(np.floor(data_num * 0.8))]
#
# # 3.2.4. split the datasets according to the splitting points
# PLIF_data_split = np.split(PLIF_data, split_points, axis=1)
#
# # 3.2.5. obtain the training, validation and testing sets
# training_PLIF_data = PLIF_data_split[0]
# validation_PLIF_data = PLIF_data_split[1]
# testing_PLIF_data = PLIF_data_split[2]
#
# training_nums = training_PLIF_data.shape[1]
# validation_nums = validation_PLIF_data.shape[1]
# testing_nums = testing_PLIF_data.shape[1]
#
# # 3.2.6. save the training, validation and testing sets
# np.save(f'data/Preprocessed_Data_old/training_PLIF_data{specified_dataset}.npy', training_PLIF_data)
# np.save(f'data/Preprocessed_Data_old/validation_PLIF_data{specified_dataset}.npy', validation_PLIF_data)
# np.save(f'data/Preprocessed_Data_old/testing_PLIF_data{specified_dataset}.npy', testing_PLIF_data)
#
# # 3.3. reshape the training, validation and testing datasets
# # get the essential shape information for reshaping datasets
# boxes = training_PLIF_data.shape[0]  # 12 boxes
#
# PLIF_height = training_PLIF_data.shape[2]
# PLIF_width = training_PLIF_data.shape[3]
#
# # then reshape datasets (i.e. flatten the 12 boxes)
# training_PLIF_data = training_PLIF_data.reshape((boxes * training_nums, PLIF_height, PLIF_width))
# validation_PLIF_data = validation_PLIF_data.reshape((boxes * validation_nums, PLIF_height, PLIF_width))
# testing_PLIF_data = testing_PLIF_data.reshape((boxes * testing_nums, PLIF_height, PLIF_width))
#
# # 3.4. obtain the training, validation and testing sets
# training_PLIF_data = np.expand_dims(training_PLIF_data, axis=1)
# validation_PLIF_data = np.expand_dims(validation_PLIF_data, axis=1)
# testing_PLIF_data = np.expand_dims(testing_PLIF_data, axis=1)
#
# # 3.5. create the corresponding datasets
# training_PLIF_dataset = MyDataset(training_PLIF_data)
# validation_PLIF_dataset = MyDataset(validation_PLIF_data)
# testing_PLIF_dataset = MyDataset(testing_PLIF_data)
#
# # 3.6. save the datasets for training
# with open(f'data/Preprocessed_Data_old/training_PLIF_dataset{specified_dataset}.pkl', 'wb') as file:
#     pickle.dump(training_PLIF_dataset, file)
#
# with open(f'data/Preprocessed_Data_old/validation_PLIF_dataset{specified_dataset}.pkl', 'wb') as file:
#     pickle.dump(validation_PLIF_dataset, file)
#
# with open(f'data/Preprocessed_Data_old/testing_PLIF_dataset{specified_dataset}.pkl', 'wb') as file:
#     pickle.dump(testing_PLIF_dataset, file)
#
# # PART 4: preprocess the datasets (for PIV)
# # 4.1. preprocess the datasets, then return the cropped datasets
# cropped_PIV_x_data, cropped_PIV_y_data, cropped_PIV_z_data = crop_old_PIVdata(files_PIV)
#
# # 4.2. get the min and max value for all PIV-x, y, z datasets
# min_PIV_x, max_PIV_x = get_min_max(cropped_PIV_x_data)
#
# # 4.3. normalize and discretize the datasets according to the min, max values
# preprocessed_PIV_x_data = preprocess_data_list(cropped_PIV_x_data, min_PIV_x, max_PIV_x)
#
# # PART 5: split the datasets to training, validation and testing datasets for training (for PIV)
# # 5.1. concatenate the datasets as required
# PIV_x_data = concatenate_data(preprocessed_PIV_x_data)
#
# # 5.2. split the datasets for training, validation and testing
# # 5.2.1. shuffle the datasets according to the same shuffled index array
# PIV_x_data = np.take(PIV_x_data, index_array, axis=1)
#
# # 5.2.2. split the datasets according to the splitting points
# PIV_x_data_split = np.split(PIV_x_data, split_points, axis=1)
#
# # 5.2.3. obtain the training, validation and testing sets
# training_x_PIV_data = PIV_x_data_split[0]
# validation_x_PIV_data = PIV_x_data_split[1]
# testing_x_PIV_data = PIV_x_data_split[2]
#
# # 5.2.4. save the training, validation and testing sets
# np.save(f'data/Preprocessed_Data_old/training_x_PIV_data{specified_dataset}.npy', training_x_PIV_data)
# np.save(f'data/Preprocessed_Data_old/validation_x_PIV_data{specified_dataset}.npy', validation_x_PIV_data)
# np.save(f'data/Preprocessed_Data_old/testing_x_PIV_data{specified_dataset}.npy', testing_x_PIV_data)
#
# # 5.3. reshape the training, validation and testing datasets
# # get the essential shape information for reshaping datasets
# PIV_x_height = training_x_PIV_data.shape[2]
# PIV_x_width = training_x_PIV_data.shape[3]
#
# # then reshape datasets (i.e. flatten the 12 boxes)
# training_x_PIV_data = training_x_PIV_data.reshape((boxes * training_nums, PIV_x_height, PIV_x_width))
# validation_x_PIV_data = validation_x_PIV_data.reshape((boxes * validation_nums, PIV_x_height, PIV_x_width))
# testing_x_PIV_data = testing_x_PIV_data.reshape((boxes * testing_nums, PIV_x_height, PIV_x_width))
#
# # 5.4. obtain the training, validation and testing sets
# training_x_PIV_data = np.expand_dims(training_x_PIV_data, axis=1)
# validation_x_PIV_data = np.expand_dims(validation_x_PIV_data, axis=1)
# testing_x_PIV_data = np.expand_dims(testing_x_PIV_data, axis=1)
#
# # 5.5. create the corresponding datasets
# training_x_PIV_dataset = MyDataset(training_x_PIV_data)
# validation_x_PIV_dataset = MyDataset(validation_x_PIV_data)
# testing_x_PIV_dataset = MyDataset(testing_x_PIV_data)
#
# # 5.6. save the datasets for training
# with open(f'data/Preprocessed_Data_old/training_x_PIV_dataset{specified_dataset}.pkl', 'wb') as file:
#     pickle.dump(training_x_PIV_dataset, file)
#
# with open(f'data/Preprocessed_Data_old/validation_x_PIV_dataset{specified_dataset}.pkl', 'wb') as file:
#     pickle.dump(validation_x_PIV_dataset, file)
#
# with open(f'data/Preprocessed_Data_old/testing_x_PIV_dataset{specified_dataset}.pkl', 'wb') as file:
#     pickle.dump(testing_x_PIV_dataset, file)

# # APPENDIX -- save the min and max value
# # try to load the existing file
# try:
#     with open('data/Preprocessed_Data_old/dataset_information.json', 'r') as file:
#         existing_data = json.load(file)
# except FileNotFoundError:
#     existing_data = {}
#
# # add new information to the file
# new_data = {
#     f'min_PLIF_dataset{specified_dataset}': float(min_PLIF),
#     f'max_PLIF_dataset{specified_dataset}': float(max_PLIF),
#     f'min_PIV_x_dataset{specified_dataset}': float(min_PIV_x),
#     f'max_PIV_x_dataset{specified_dataset}': float(max_PIV_x)
# }
#
# existing_data.update(new_data)
#
# # save the updated data information
# with open('data/Preprocessed_Data_old/dataset_information.json', 'w') as file:
#     json.dump(existing_data, file)
