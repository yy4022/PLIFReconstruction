import os.path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import summary

from fullyCNN.train import train_epoch
from fullyCNN.validate import validate_epoch
from preprocess_dataset import preprocess_data, show_image, MyDataset, \
    show_box_PIV, show_box_PLIF, crop_old_PIVdata, crop_old_PLIFdata, get_min_max, preprocess_data_list
from fullyCNN.neural_net import FullyCNN
from result_visualiser import show_loss

"""
This file is used for testing the whole process of training the global model.
NOTE: there is no need to use the testing datasets during the training process.
"""

# PART 1: define the parameters
# 1. choose the device
torch.cuda.set_device(0)

print(torch.__version__)
print(torch.cuda.is_available())
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Selected device: {device}")

# 2. define the parameters for training the model
batch_size = 100
rows = 3
columns = 4
img_num = 0
EPOCHS = 1000
lr = 0.0001
if_existing = False # a flag recording if there is an existing fullyCNN model

# 3. provide filenames of PIV, PLIF data
files_PIV = ['data/Attached state/D1F1_air240_PIV_1001to2000.mat',
             'data/Attached state/D1F1_air240_PIV_2001to3000.mat']

files_PLIF = ['data/Attached state/D1F1_air240_PLIF_1001to2000.mat',
              'data/Attached state/D1F1_air240_PLIF_2001to3000.mat']

# PART 2: preprocess the datasets
# 1. preprocess the datasets, then return the cropped datasets
cropped_PLIF_data = crop_old_PLIFdata(files_PLIF)
cropped_PIV_x_data, cropped_PIV_y_data, cropped_PIV_z_data = crop_old_PIVdata(files_PIV)

# 2. get the min and max value for all PIV-x, y, z and PLIF datasets
min_PLIF, max_PLIF = get_min_max(cropped_PLIF_data)
min_PIV_x, max_PIV_x = get_min_max(cropped_PIV_x_data)
min_PIV_y, max_PIV_y = get_min_max(cropped_PIV_y_data)
min_PIV_z, max_PIV_z = get_min_max(cropped_PIV_z_data)

# 3. normalize and discretize the datasets according to the min, max values
PLIF_data = preprocess_data_list(cropped_PLIF_data, min_PLIF, max_PLIF)
PIV_x_data = preprocess_data_list(cropped_PIV_x_data, min_PIV_x, max_PIV_x)
PIV_y_data = preprocess_data_list(cropped_PIV_y_data, min_PIV_y, max_PIV_y)
PIV_z_data = preprocess_data_list(cropped_PIV_z_data, min_PIV_z, max_PIV_z)

# PLIF_data1 = preprocess_old_data(cropped_PLIF_data1, min_PLIF, max_PLIF)
# PLIF_data2 = preprocess_old_data(cropped_PLIF_data2, min_PLIF, max_PLIF)
#
# PIV_x_data1 = preprocess_old_data(cropped_PIV_data1[0, :, :, :], min_PIV_x, max_PIV_x)
# PIV_x_data2 = preprocess_old_data(cropped_PIV_data2[0, :, :, :], min_PIV_x, max_PIV_x)
#
# PIV_y_data1 = preprocess_old_data(cropped_PIV_data1[1, :, :, :], min_PIV_y, max_PIV_y)
# PIV_y_data2 = preprocess_old_data(cropped_PIV_data2[1, :, :, :], min_PIV_y, max_PIV_y)
#
# PIV_z_data1 = preprocess_old_data(cropped_PIV_data1[2, :, :, :], min_PIV_z, max_PIV_z)
# PIV_z_data2 = preprocess_old_data(cropped_PIV_data2[2, :, :, :], min_PIV_z, max_PIV_z)
#
# # 2. concatenate the datasets as required
# PIV_x_attached_data = np.concatenate((PIV_x_data1, PIV_x_data2), axis=1) # attached-2000
# PLIF_attached_data = np.concatenate((PLIF_data1, PLIF_data2), axis=1)
#
# # 3. split the datasets for training, validation and testing
# # 3.1. get the total number of data
# attached_num = PIV_x_attached_data.shape[1]
#
# # 3.2. calculate the position of splitting points to satisfy the proportion (3: 1: 1)
# attached_split_points = [int(np.floor(attached_num * 0.6)), int(np.floor(attached_num * 0.8))]
#
# # 3.3. split the datasets according to the splitting points
# PIV_x_attached_data_split = np.split(PIV_x_attached_data, attached_split_points, axis=1)
# PLIF_attached_data_split = np.split(PLIF_attached_data, attached_split_points, axis=1)
#
# # 3.4. obtain the training, validation, testing sets
# training_x_PIV_data = PIV_x_attached_data_split[0]
# training_PLIF_data = PLIF_attached_data_split[0]
# training_nums = training_PLIF_data.shape[1]
#
# validation_x_PIV_data = PIV_x_attached_data_split[1]
# validation_PLIF_data = PLIF_attached_data_split[1]
# validation_nums = validation_PLIF_data.shape[1]
#
# testing_x_PIV_data = PIV_x_attached_data_split[2]
# testing_PLIF_data = PLIF_attached_data_split[2]
# testing_nums = testing_PLIF_data.shape[1]
#
# # 3.5. reshape the training, validation, testing datasets
# # get the essential shape information for reshaping datasets
# boxes = training_x_PIV_data.shape[0] # 12 boxes
#
# PIV_x_height = training_x_PIV_data.shape[2]
# PIV_x_width = training_x_PIV_data.shape[3]
#
# PLIF_height = training_PLIF_data.shape[2]
# PLIF_width = training_PLIF_data.shape[3]
#
# # then reshape datasets (i.e. flatten the 12 boxes)
# training_x_PIV_data = training_x_PIV_data.reshape((boxes * training_nums, PIV_x_height, PIV_x_width))
# training_PLIF_data = training_PLIF_data.reshape((boxes * training_nums, PLIF_height, PLIF_width))
#
# validation_x_PIV_data = validation_x_PIV_data.reshape((boxes * validation_nums, PIV_x_height, PIV_x_width))
# validation_PLIF_data = validation_PLIF_data.reshape((boxes * validation_nums, PLIF_height, PLIF_width))
#
# # 3.6. obtain the training, validation, training sets
# training_x_PIV_data = np.expand_dims(training_x_PIV_data, axis=1)
# training_PLIF_data = np.expand_dims(training_PLIF_data, axis=1)
#
# validation_x_PIV_data = np.expand_dims(validation_x_PIV_data, axis=1)
# validation_PLIF_data = np.expand_dims(validation_PLIF_data, axis=1)
#
# # 4.5. create the corresponding datasets
# training_x_PIV_dataset = MyDataset(training_x_PIV_data)
# training_PLIF_dataset = MyDataset(training_PLIF_data)
#
# validation_x_PIV_dataset = MyDataset(validation_x_PIV_data)
# validation_PLIF_dataset = MyDataset(validation_PLIF_data)
#
# # 4.6. create the corresponding dataloaders
# training_x_PIV_loader = DataLoader(dataset=training_x_PIV_dataset, batch_size=batch_size, shuffle=False)
# training_PLIF_loader = DataLoader(dataset=training_PLIF_dataset, batch_size=batch_size, shuffle=False)
#
# validation_x_PIV_loader = DataLoader(dataset=validation_x_PIV_dataset, batch_size=batch_size, shuffle=False)
# validation_PLIF_loader = DataLoader(dataset=validation_PLIF_dataset, batch_size=batch_size, shuffle=False)
#
# # PART 3: preparation before training the model
# # 1. define the FullyCNN model
# fullyCNN = FullyCNN()
#
# # check if there is an existing model
# if os.path.exists('./model/fullyCNN.pt'):
#     fullyCNN = torch.load('./model/fullyCNN.pt')
#
#     # set the if_existing flag
#     if_existing = True
#     print("Load the existing fullyCNN model, then continue training.")
# else:
#     print("No existing fullyCNN model, so create a new one.")
#
# fullyCNN = fullyCNN.to(device)
# input_shape = (1, 124, 93)
# summary(fullyCNN, input_shape)
#
# # 2. create a numpy array for recording the loss,
# #   and set the best (validation) loss for updating the model
# train_loss_records = []
# validation_loss_records = []
#
# train_loss_records = np.array(train_loss_records)
# validation_loss_records = np.array(validation_loss_records)
#
# best_loss = 10.0
#
# if if_existing == True:
#     train_loss_records = \
#         np.append(train_loss_records, np.load('./result/train_loss_records.npy'))
#
#     validation_loss_records = \
#         np.append(validation_loss_records, np.load('./result/validation_loss_records.npy'))
#
#     best_loss = validation_loss_records.min()
#     print(f"Load the existing loss records, and current best loss is {best_loss}.")
#
# else:
#     print("No existing loss records, start recording from the beginning.")
#
# # 3. define the loss function and the optimizer
# loss_fn = nn.MSELoss()
# torch.manual_seed(0)
#
# optimizer = torch.optim.Adam(fullyCNN.parameters(), lr=lr, weight_decay=1e-05)
#
# # PART 4: the looping process of training the model
# # NOTE: the test file takes PIV-x (dimension-0) as an example
# for epoch in range(EPOCHS):
#
#     train_loss = train_epoch(fullyCNN=fullyCNN, device=device, dataloader_in=training_PLIF_loader,
#                              dataloader_out=training_x_PIV_loader, loss_fn=loss_fn, optimizer=optimizer)
#
#     validation_loss = validate_epoch(fullyCNN=fullyCNN, device=device, dataloader_in=validation_PLIF_loader,
#                                      dataloader_out=validation_x_PIV_loader, loss_fn=loss_fn)
#
#     print(
#         '\n EPOCH {}/{} \t train loss {} \t validate loss {}'.format(epoch + 1, EPOCHS, train_loss,
#                                                                      validation_loss))
#
#     train_loss_records = np.append(train_loss_records, train_loss)
#     validation_loss_records = np.append(validation_loss_records, validation_loss)
#
#     if validation_loss < best_loss:
#         best_loss = validation_loss
#         torch.save(fullyCNN, './model/fullyCNN.pt')
#
# # save loss records of training and validation process
# np.save("./result/train_loss_records.npy", train_loss_records)
# np.save("./result/validation_loss_records.npy", validation_loss_records)
#
# loss_records = {
#     'train_loss_records': train_loss_records,
#     'validation_loss_records': validation_loss_records
# }
#
# # PART 5: show the results
# # 5.1. show the loss records of the whole training process
# show_loss(loss_records, "fullyCNN_loss.png")
