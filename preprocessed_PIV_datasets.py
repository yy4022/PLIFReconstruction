import json

import numpy as np

from methods_preprocess import get_min_max, crop_old_PIVdata, concatenate_data, preprocess_data_list

"""
SECTION 1: This section is used for getting the global min and max values of PIV dataset for normalization.
(NOTE:  1. make sure that there is no dataset_information.json file in the directory
        2. you can comment this section after implementing it)
"""

# # PART 1: define the parameters for training the model
# # provide the filename of PIV data
# files_PIV = [
#     'data/Full dataset/D1F1_air240_PIV_1to14999.mat',
# ]
#
# # PART 2: get the min and max value for this specified dataset (for PIV)
# # 2.1. preprocess the datasets, then return the cropped datasets
# cropped_PIV_x_data, cropped_PIV_y_data, cropped_PIV_z_data = crop_old_PIVdata(files_PIV)
#
# # 2.2. get the min and max value for this PIV-x, y, z dataset
# min_PIV_x, max_PIV_x = get_min_max(cropped_PIV_x_data)
# min_PIV_y, max_PIV_y = get_min_max(cropped_PIV_y_data)
# min_PIV_z, max_PIV_z = get_min_max(cropped_PIV_z_data)
#
# # PART 3: compare the min and max value with the saved one
# existing_data = {}
#
# # add new information to the file
# new_data = {
#     'min_PIV_x': float(min_PIV_x),
#     'max_PIV_x': float(max_PIV_x),
#     'min_PIV_y': float(min_PIV_y),
#     'max_PIV_y': float(max_PIV_y),
#     'min_PIV_z': float(min_PIV_z),
#     'max_PIV_z': float(max_PIV_z),
# }
#
# existing_data.update(new_data)
#
# print(existing_data)
#
# # 3.2. save the updated data information
# with open('data/Preprocessed_Data_Fulldataset/dataset_information_PIV.json', 'w') as file:
#     json.dump(existing_data, file)

"""
SECTION 2: This section is used for cropping, normalizing and discretizing all PIV datasets.
NOTE: 1. the result data would be stored in the same directory.
      2. the whole dataset will be split for getting the unified format with PLIF datasets.
"""

# PART 1: provide the essential information
# 1.1. define the parameters
dataset_nums = 8

# 1.2. provide the filename of PIV data
files_PIV = [
    'data/Full dataset/D1F1_air240_PIV_1to14999.mat',
]

# PART 2: get the global min and max value
with open('data/Preprocessed_Data_Fulldataset/dataset_information_PIV.json', 'r') as file:
    existing_data = json.load(file)

min_PIV_x = np.float32(existing_data['min_PIV_x'])
max_PIV_x = np.float32(existing_data['max_PIV_x'])
min_PIV_y = np.float32(existing_data['min_PIV_y'])
max_PIV_y = np.float32(existing_data['max_PIV_y'])
min_PIV_z = np.float32(existing_data['min_PIV_z'])
max_PIV_z = np.float32(existing_data['max_PIV_z'])

# PART 3:  preprocess the datasets (crop, normalize and discretize)
cropped_PIV_x_data, cropped_PIV_y_data, cropped_PIV_z_data = crop_old_PIVdata(files_PIV)

# normalize and discretize the datasets according to the min, max values
preprocessed_PIV_x_data = preprocess_data_list(cropped_PIV_x_data, min_PIV_x, max_PIV_x)
preprocessed_PIV_y_data = preprocess_data_list(cropped_PIV_y_data, min_PIV_y, max_PIV_y)
preprocessed_PIV_z_data = preprocess_data_list(cropped_PIV_z_data, min_PIV_z, max_PIV_z)

PIV_x_data = concatenate_data(preprocessed_PIV_x_data)
PIV_y_data = concatenate_data(preprocessed_PIV_y_data)
PIV_z_data = concatenate_data(preprocessed_PIV_z_data)

# PART 4: save the specified dataset
for i in range(dataset_nums):
    start_num = i * 2000
    end_num = start_num + 2000

    if end_num > PIV_x_data.shape[1]:
        end_num = PIV_x_data.shape[1]

    PIV_x_data_i = PIV_x_data[:, start_num:end_num, :, :]
    PIV_y_data_i = PIV_y_data[:, start_num:end_num, :, :]
    PIV_z_data_i = PIV_z_data[:, start_num:end_num, :, :]

    np.save(f'data/Full dataset/PIV_x_dataset{i + 1}.npy', PIV_x_data_i)
    np.save(f'data/Full dataset/PIV_y_dataset{i + 1}.npy', PIV_y_data_i)
    np.save(f'data/Full dataset/PIV_z_dataset{i + 1}.npy', PIV_z_data_i)
