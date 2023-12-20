import json

import numpy as np

from methods_preprocess import crop_old_PLIFdata, get_min_max, preprocess_data_list, concatenate_data

"""
SECTION 1: This section is used for getting the global min and max values of PLIF dataset for normalization.
(NOTE:  1. make sure that there is no dataset_information.json file in the directory
        2. you can comment this section after implementing it)
"""

# # PART 1: define the parameters for training the model
# # provide the filename of PLIF data
# files_PLIF = [
#     # 'data/PLIF240air/D1F1_air240_PLIF_1to2000.mat',  # training + validation - dataset1 (2000)
#     # 'data/PLIF240air/D1F1_air240_PLIF_2001to4000.mat',  # training + validation - dataset2 (2000)
#     # 'data/PLIF240air/D1F1_air240_PLIF_4001to6000.mat',  # training + validation - dataset3 (2000)
#     # 'data/PLIF240air/D1F1_air240_PLIF_6001to8000.mat',  # training + validation - dataset4 (2000)
#     # 'data/PLIF240air/D1F1_air240_PLIF_8001to10000.mat',  # training + validation - dataset5 (2000)
#     # 'data/PLIF240air/D1F1_air240_PLIF_10001to12000.mat',  # training + validation - dataset6 (2000)
#     # 'data/PLIF240air/D1F1_air240_PLIF_12001to14000.mat',  # training + validation - dataset7 (2000)
#     'data/PLIF240air/D1F1_air240_PLIF_14001to14999.mat',  # testing - dataset8 (999)
# ]
#
# # PART 2: get the min and max value for this specified dataset (for PLIF)
# # 2.1. preprocess the datasets, then return the cropped datasets
# cropped_PLIF_data = crop_old_PLIFdata(files_PLIF)
#
# # 2.2. get the min and max value for this PLIF dataset
# min_PLIF, max_PLIF = get_min_max(cropped_PLIF_data)
#
# # PART 3: compare the min and max value with the saved one
# # 3.1. try to load the existing file
# try:
#     # if the values have existed, compare and update the value
#     with open('data/Preprocessed_Data_Fulldataset/dataset_information_PLIF.json', 'r') as file:
#         existing_data = json.load(file)
#
#     current_min_PLIF = existing_data['min_PLIF']
#     current_max_PLIF = existing_data['max_PLIF']
#
#     if current_min_PLIF > float(min_PLIF):
#         existing_data['min_PLIF'] = float(min_PLIF)
#     if current_max_PLIF < float(min_PLIF):
#         existing_data['max_PLIF'] = float(min_PLIF)
#
# except FileNotFoundError:
#     # if the values have not existed, create a new one
#     existing_data = {}
#
#     # add new information to the file
#     new_data = {
#         'min_PLIF': float(min_PLIF),
#         'max_PLIF': float(max_PLIF),
#     }
#
#     existing_data.update(new_data)
#
# print(existing_data)
#
# # 3.2. save the updated data information
# with open('data/Preprocessed_Data_Fulldataset/dataset_information_PLIF.json', 'w') as file:
#     json.dump(existing_data, file)

"""
SECTION 2: This section is used for cropping, normalizing and discretizing all PLIF datasets.
NOTE: the result data would be stored in the same directory.
"""

# PART 1: provide the essential information
# 1.1. define the parameters
specified_dataset = 8

# 1.2. provide the filename of PLIF data
files_PLIF = [
    # 'data/PLIF240air/D1F1_air240_PLIF_1to2000.mat',  # training + validation - dataset1 (2000)
    # 'data/PLIF240air/D1F1_air240_PLIF_2001to4000.mat',  # training + validation - dataset2 (2000)
    # 'data/PLIF240air/D1F1_air240_PLIF_4001to6000.mat',  # training + validation - dataset3 (2000)
    # 'data/PLIF240air/D1F1_air240_PLIF_6001to8000.mat',  # training + validation - dataset4 (2000)
    # 'data/PLIF240air/D1F1_air240_PLIF_8001to10000.mat',  # training + validation - dataset5 (2000)
    # 'data/PLIF240air/D1F1_air240_PLIF_10001to12000.mat',  # training + validation - dataset6 (2000)
    # 'data/PLIF240air/D1F1_air240_PLIF_12001to14000.mat',  # training + validation - dataset7 (2000)
    'data/PLIF240air/D1F1_air240_PLIF_14001to14999.mat',  # testing - dataset8 (999)
]

# PART 2: get the global min and max value
with open('data/Preprocessed_Data_Fulldataset/dataset_information_PLIF.json', 'r') as file:
    existing_data = json.load(file)

min_PLIF = np.float32(existing_data['min_PLIF'])
max_PLIF = np.float32(existing_data['max_PLIF'])

# PART 3:  preprocess the datasets (crop, normalize and discretize)
cropped_PLIF_data = crop_old_PLIFdata(files_PLIF)

# normalize and discretize the datasets according to the min, max values
preprocessed_PLIF_data = preprocess_data_list(cropped_PLIF_data, min_PLIF, max_PLIF)

# concatenate the datasets as required
PLIF_data = concatenate_data(preprocessed_PLIF_data)

# PART 4: save this specified data
np.save(f'data/PLIF240air/PLIF_dataset{specified_dataset}.npy', PLIF_data)
