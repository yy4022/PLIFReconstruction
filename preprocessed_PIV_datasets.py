import json

from methods_preprocess import get_min_max, crop_old_PIVdata

"""
SECTION 1: This section is used for getting the global min and max values of PIV dataset for normalization.
(NOTE:  1. make sure that there is no dataset_information.json file in the directory
        2. you can comment this section after implementing it)
"""

# PART 1: define the parameters for training the model
# provide the filename of PIV data
files_PIV = [
    'data/Full dataset/D1F1_air240_PIV_1to14999.mat',
]

# PART 2: get the min and max value for this specified dataset (for PIV)
# 2.1. preprocess the datasets, then return the cropped datasets
cropped_PIV_x_data, cropped_PIV_y_data, cropped_PIV_z_data = crop_old_PIVdata(files_PIV)

# 2.2. get the min and max value for this PIV-x, y, z dataset
min_PIV_x, max_PIV_x = get_min_max(cropped_PIV_x_data)
min_PIV_y, max_PIV_y = get_min_max(cropped_PIV_y_data)
min_PIV_z, max_PIV_z = get_min_max(cropped_PIV_z_data)

# PART 3: compare the min and max value with the saved one
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
