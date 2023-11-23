import numpy as np
import torch
from torch.utils.data import DataLoader
from torchsummary import summary

from preprocess_dataset import preprocess_data, preprocess_old_data, show_image, MyDataset, \
                               show_box_PIV, show_box_PLIF
from globalCNN.neural_net import FullyCNN

# PART 1: define the parameters
# 1. choose the device
torch.cuda.set_device(0)

print(torch.__version__)
print(torch.cuda.is_available())
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Selected device: {device}")

# 2. define the parameters for training the model
batch_size = 20
rows = 3
columns = 4
img_num = 99

# # 3. provide filenames of PIV, PLIF data
# file1_PIV = str('data/Attached state/D1F1_air240_PIV_1001to2000.mat') # attached-1000
# file1_PLIF = str('data/Attached state/D1F1_air240_PLIF_1001to2000.mat')
# # file2_PIV = str('data/Attached state/D1F1_air240_PIV_2001to3000.mat') # attached-1000
# # file2_PLIF = str('data/Attached state/D1F1_air240_PLIF_2001to3000.mat')
#
# # PART 2: preprocess the datasets
# # 1. preprocess the datasets, then return the cropped datasets
# PIV_data1, PLIF_data1 = preprocess_old_data(file1_PIV, file1_PLIF)[:2]
# # PIV_data2, PLIF_data2 = preprocess_old_data(file2_PIV, file2_PLIF)[:2]
#
# print(np.shape(PIV_data1))
# print(np.shape(PLIF_data1))
#
# print(np.shape(PIV_data1[:, :, img_num, :, :]))
# print(np.shape(PLIF_data1[:, img_num, :, :]))

# show_box_PIV(PIV_data1[:, :, img_num, :, :], dimension=1, rows=rows, columns=columns) # x-axis
# show_box_PIV(PIV_data1[:, :, img_num, :, :], dimension=2, rows=rows, columns=columns) # y-axis
# show_box_PIV(PIV_data1[:, :, img_num, :, :], dimension=3, rows=rows, columns=columns) # z-axis
# show_box_PLIF(PLIF_data1[:, img_num, :, :], rows=rows, columns=columns)

# define the FullyCNN model
fullyCNN = FullyCNN()
fullyCNN = fullyCNN.to(device)
input_shape = (1, 124, 93)
summary(fullyCNN, input_shape)

# # 2. concatenate the datasets as required
# PIV_attached_data = np.concatenate((PIV_data1, PIV_data2), axis=1) # attached-2000
# PLIF_attached_data = np.concatenate((PLIF_data1, PLIF_data2), axis=0)
#
# # 3. split the datasets for training, evaluation and testing
# # 3.1. get the total number of data
# attached_num = PIV_attached_data.shape[1]
#
# # 3.2. calculate the position of splitting points to satisfy the proportion (3: 1: 1)
# attached_split_points = [int(np.floor(attached_num * 0.6)), int(np.floor(attached_num * 0.8))]
#
# # 3.3. split the datasets according to the splitting points
# PIV_attached_data_split = np.split(PIV_attached_data, attached_split_points, axis=1)
# PLIF_attached_data_split = np.split(PLIF_attached_data, attached_split_points, axis=0)
#
# # 3.4. obtain the training, validation, training sets
# training_PIV_data = PIV_attached_data_split[0]
# training_PLIF_data = PLIF_attached_data_split[0]
#
# validation_PIV_data = PIV_attached_data_split[1]
# validation_PLIF_data = PLIF_attached_data_split[1]
#
# testing_PIV_data = PIV_attached_data_split[2]
# testing_PLIF_data = PLIF_attached_data_split[2]

# # 3.5. create the corresponding datasets
# training_PIV_data = MyDataset(training_PIV_data)
# training_PLIF_data = MyDataset(training_PLIF_data)
#
# validation_PIV_data = MyDataset(validation_PIV_data)
# validation_PLIF_data = MyDataset(validation_PLIF_data)
#
# testing_PIV_data = MyDataset(testing_PIV_data)
# testing_PLIF_data = MyDataset(testing_PLIF_data)
#
# # 3.6. create the corresponding dataloaders
# training_PIV_loader = DataLoader(dataset=training_PIV_data, batch_size=batch_size, shuffle=False)
# training_PLIF_loader = DataLoader(dataset=training_PLIF_data, batch_size=batch_size, shuffle=False)
#
# validation_PIV_loader = DataLoader(dataset=validation_PIV_data, batch_size=batch_size, shuffle=False)
# validation_PLIF_loader = DataLoader(dataset=validation_PLIF_data, batch_size=batch_size, shuffle=False)
#
# testing_PIV_loader = DataLoader(dataset=testing_PIV_data, batch_size=batch_size, shuffle=False)
# testing_PLIF_loader = DataLoader(dataset=testing_PLIF_data, batch_size=batch_size, shuffle=False)

