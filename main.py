import numpy as np
import torch

from preprocess_dataset import preprocess_data, preprocess_old_data, show_image, MyDataset

# PART 1: define the parameters
# 1. choose the device
print(torch.__version__)
print(torch.cuda.is_available())
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Selected device: {device}")

# 2. provide all filenames of PIV, PLIF data
file1_PIV = str('data/Attached state/D1F1_air240_PIV_1001to2000.mat') # attached-1000
file1_PLIF = str('data/Attached state/D1F1_air240_PLIF_1001to2000.mat')
file2_PIV = str('data/Attached state/D1F1_air240_PIV_2001to3000.mat') # attached-1000
file2_PLIF = str('data/Attached state/D1F1_air240_PLIF_2001to3000.mat')

# file3_PIV = str('data/Detachment process/D1F1_air240_PIV_4201to4500.mat') # detachment-300
# file3_PLIF = str('data/Detachment process/D1F1_air240_PLIF_4201to4500.mat')
file4_PIV = str('data/Detachment process/D1F1_air240_PIV_13401to13600.mat') # detachment-200
file4_PLIF = str('data/Detachment process/D1F1_air240_PLIF_13401to13600.mat')

file5_PIV = str('data/Lifted state/D1F1_air240_PIV_13601to14000.mat') # lift-400
file5_PLIF = str('data/Lifted state/D1F1_air240_PLIF_13601to14000.mat')
file6_PIV = str('data/Lifted state/D1F1_air240_PIV_14001to14999.mat') # lift-999
file6_PLIF = str('data/Lifted state/D1F1_air240_PLIF_14001to14999.mat')

file7_PIV = str('data/Reattachment process/D1F1_air240_PIV_451to650.mat') # reattachment-200
file7_PLIF = str('data/Reattachment process/D1F1_air240_PLIF_451to650.mat')
file8_PIV = str('data/Reattachment process/D1F1_air240_PIV_6201to6700.mat') # reattachment-500
file8_PLIF = str('data/Reattachment process/D1F1_air240_PLIF_6201to6700.mat')

# PART 2: preprocess the datasets
# 1. preprocess the datasets, then return the cropped datasets
PIV_data1, PLIF_data1 = preprocess_old_data(file1_PIV, file1_PLIF)[:2]
PIV_data2, PLIF_data2 = preprocess_old_data(file2_PIV, file2_PLIF)[:2]
# PIV_data3, PLIF_data3 = preprocess_old_data(file3_PIV, file3_PLIF)[:2]
PIV_data4, PLIF_data4 = preprocess_old_data(file4_PIV, file4_PLIF)[:2]
PIV_data5, PLIF_data5 = preprocess_old_data(file5_PIV, file5_PLIF)[:2]
PIV_data6, PLIF_data6 = preprocess_old_data(file6_PIV, file6_PLIF)[:2]
PIV_data7, PLIF_data7 = preprocess_old_data(file7_PIV, file7_PLIF)[:2]
PIV_data8, PLIF_data8 = preprocess_old_data(file8_PIV, file8_PLIF)[:2]

# 2. concatenate the datasets as required
PIV_attached_data = np.concatenate((PIV_data1, PIV_data2), axis=1) # attached-2000
PLIF_attached_data = np.concatenate((PLIF_data1, PLIF_data2), axis=0)

PIV_detachment_data = PIV_data4 # detachment-200
PLIF_detachment_data = PLIF_data4

PIV_lifted_data = np.concatenate((PIV_data5, PIV_data6), axis=1) # lifted-1399
PLIF_lifted_data = np.concatenate((PLIF_data5, PLIF_data6), axis=0)

PIV_reattachment_data = np.concatenate((PIV_data7, PIV_data8), axis=1) # reattachment-700
PLIF_reattachment_data = np.concatenate((PLIF_data7, PLIF_data8), axis=0)

# 3. split the datasets for training, evaluation and testing
# 3.1. get the total number of data for every state
attached_num = PIV_attached_data.shape[1]
detachment_num = PIV_detachment_data.shape[1]
lifted_num = PIV_lifted_data.shape[1]
reattachment_num = PIV_reattachment_data.shape[1]

# 3.2. calculate the position of splitting points to satisfy the proportion (3: 1: 1)
attached_split_points = [int(np.floor(attached_num * 0.6)), int(np.floor(attached_num * 0.8))]
detachment_split_points = [int(np.floor(detachment_num * 0.6)), int(np.floor(detachment_num * 0.8))]
lifted_split_points = [int(np.floor(lifted_num * 0.6)), int(np.floor(lifted_num * 0.8))]
reattachment_split_points = [int(np.floor(reattachment_num * 0.6)), int(np.floor(reattachment_num * 0.8))]

# 3.3. split the datasets according to the splitting points
PIV_attached_data_split = np.split(PIV_attached_data, attached_split_points, axis=1)
PLIF_attached_data_split = np.split(PLIF_attached_data, attached_split_points, axis=0)

PIV_detachment_data_split = np.split(PIV_detachment_data, detachment_split_points, axis=1)
PLIF_detachment_data_split = np.split(PLIF_detachment_data, detachment_split_points, axis=0)

PIV_lifted_data_split = np.split(PIV_lifted_data, lifted_split_points, axis=1)
PLIF_lifted_data_split = np.split(PLIF_lifted_data, lifted_split_points, axis=0)

PIV_reattachment_data_split = np.split(PIV_reattachment_data, reattachment_split_points, axis=1)
PLIF_reattachment_data_split = np.split(PLIF_reattachment_data, reattachment_split_points, axis=0)

# 3.4. obtain the training, validation, training sets
training_PIV_data = np.concatenate((PIV_attached_data_split[0], PIV_detachment_data_split[0],
                                    PIV_lifted_data_split[0], PIV_reattachment_data_split[0]), axis=1)
training_PLIF_data = np.concatenate((PLIF_attached_data_split[0], PLIF_detachment_data_split[0],
                                     PLIF_lifted_data_split[0], PLIF_reattachment_data_split[0]), axis=0)

validation_PIV_data = np.concatenate((PIV_attached_data_split[1], PIV_detachment_data_split[1],
                                      PIV_lifted_data_split[1], PIV_reattachment_data_split[1]), axis=1)
validation_PLIF_data = np.concatenate((PLIF_attached_data_split[1], PLIF_detachment_data_split[1],
                                       PLIF_lifted_data_split[1], PLIF_reattachment_data_split[1]), axis=0)

testing_PIV_data = np.concatenate((PIV_attached_data_split[2], PIV_detachment_data_split[2],
                                   PIV_lifted_data_split[2], PIV_reattachment_data_split[2]), axis=1)
testing_PLIF_data = np.concatenate((PLIF_attached_data_split[2], PLIF_detachment_data_split[2],
                                    PLIF_lifted_data_split[2], PLIF_reattachment_data_split[2]), axis=0)

# 3.5. create the corresponding datasets
training_PIV_data = MyDataset(training_PIV_data)
training_PLIF_data = MyDataset(training_PLIF_data)

validation_PIV_data = MyDataset(validation_PIV_data)
validation_PLIF_data = MyDataset(validation_PLIF_data)

testing_PIV_data = MyDataset(testing_PIV_data)
testing_PLIF_data = MyDataset(testing_PLIF_data)


