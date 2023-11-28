import os.path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import summary

from globalCNN.predict import fullyCNN_predict
from globalCNN.train import train_epoch
from globalCNN.validate import validate_epoch
from preprocess_dataset import preprocess_data, preprocess_old_data, show_image, MyDataset, \
    show_box_PIV, show_box_PLIF, crop_old_data
from globalCNN.neural_net import FullyCNN
from result_visualiser import show_loss, show_difference, show_comparison

"""
This file is used for showing the results via the trained model.
NOTE: all datasets used in this file should not be shuffled.
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
img_num = 99
EPOCHS = 1000
lr = 0.0001
if_existing = False # a flag recording if there is an existing fullyCNN model

# 3. provide filenames of PIV, PLIF data
file1_PIV = str('data/Attached state/D1F1_air240_PIV_1001to2000.mat') # attached-1000
file1_PLIF = str('data/Attached state/D1F1_air240_PLIF_1001to2000.mat')
file2_PIV = str('data/Attached state/D1F1_air240_PIV_2001to3000.mat') # attached-1000
file2_PLIF = str('data/Attached state/D1F1_air240_PLIF_2001to3000.mat')

# PART 2: preprocess the datasets
# 1. preprocess the datasets, then return the cropped datasets
cropped_PIV_data1, cropped_PLIF_data1 = crop_old_data(file1_PIV, file1_PLIF)
cropped_PIV_data2, cropped_PLIF_data2 = crop_old_data(file2_PIV, file2_PLIF)

# 2. get the min and max value for all PIV-x, y, z and PLIF datasets
min_PLIF = min(np.amin(cropped_PLIF_data1), np.amin(cropped_PLIF_data2))
max_PLIF = max(np.amax(cropped_PLIF_data1), np.amax(cropped_PLIF_data2))

min_PIV_x = min(np.amin(cropped_PIV_data1[0, :, :, :]), np.amin(cropped_PIV_data2[0, :, :, :]))
max_PIV_x = max(np.amax(cropped_PIV_data1[0, :, :, :]), np.amax(cropped_PIV_data2[0, :, :, :]))

min_PIV_y = min(np.amin(cropped_PIV_data1[1, :, :, :]), np.amin(cropped_PIV_data2[1, :, :, :]))
max_PIV_y = max(np.amax(cropped_PIV_data1[1, :, :, :]), np.amax(cropped_PIV_data2[1, :, :, :]))

min_PIV_z = min(np.amin(cropped_PIV_data1[2, :, :, :]), np.amin(cropped_PIV_data2[2, :, :, :]))
max_PIV_z = max(np.amax(cropped_PIV_data1[2, :, :, :]), np.amax(cropped_PIV_data2[2, :, :, :]))

# 3. normalize and discretize the datasets according to the min, max values
PLIF_data1 = preprocess_old_data(cropped_PLIF_data1, min_PLIF, max_PLIF)
PLIF_data2 = preprocess_old_data(cropped_PLIF_data2, min_PLIF, max_PLIF)

PIV_x_data1 = preprocess_old_data(cropped_PIV_data1[0, :, :, :], min_PIV_x, max_PIV_x)
PIV_x_data2 = preprocess_old_data(cropped_PIV_data2[0, :, :, :], min_PIV_x, max_PIV_x)

PIV_y_data1 = preprocess_old_data(cropped_PIV_data1[1, :, :, :], min_PIV_y, max_PIV_y)
PIV_y_data2 = preprocess_old_data(cropped_PIV_data2[1, :, :, :], min_PIV_y, max_PIV_y)

PIV_z_data1 = preprocess_old_data(cropped_PIV_data1[2, :, :, :], min_PIV_z, max_PIV_z)
PIV_z_data2 = preprocess_old_data(cropped_PIV_data2[2, :, :, :], min_PIV_z, max_PIV_z)

# print(np.shape(PLIF_data1))
# print(np.shape(PIV_x_data1))

# print(min_PLIF, min_PIV_x, min_PIV_y, min_PIV_z)
# print(max_PLIF, max_PIV_x, max_PIV_y, max_PIV_z)

# show_box_PIV(PIV_data1[:, :, img_num, :, :], dimension=3, rows=rows, columns=columns) # z-axis
# show_box_PLIF(PLIF_data1[:, img_num, :, :], rows=rows, columns=columns)

# 2. concatenate the datasets as required
PIV_x_attached_data = np.concatenate((PIV_x_data1, PIV_x_data2), axis=1) # attached-2000
PLIF_attached_data = np.concatenate((PLIF_data1, PLIF_data2), axis=1)

# 3. split the datasets for training, validation and testing
# 3.1. get the total number of data
attached_num = PIV_x_attached_data.shape[1]

# 3.2. calculate the position of splitting points to satisfy the proportion (3: 1: 1)
attached_split_points = [int(np.floor(attached_num * 0.6)), int(np.floor(attached_num * 0.8))]

# 3.3. split the datasets according to the splitting points
PIV_x_attached_data_split = np.split(PIV_x_attached_data, attached_split_points, axis=1)
PLIF_attached_data_split = np.split(PLIF_attached_data, attached_split_points, axis=1)

# 3.4. obtain the training, validation, testing sets
training_x_PIV_data = PIV_x_attached_data_split[0]
training_PLIF_data = PLIF_attached_data_split[0]
training_nums = training_PLIF_data.shape[1]

validation_x_PIV_data = PIV_x_attached_data_split[1]
validation_PLIF_data = PLIF_attached_data_split[1]
validation_nums = validation_PLIF_data.shape[1]

testing_x_PIV_data = PIV_x_attached_data_split[2]
testing_PLIF_data = PLIF_attached_data_split[2]
testing_nums = testing_PLIF_data.shape[1]

# 3.5. reshape the training, validation, testing datasets
# get the essential shape information for reshaping datasets
boxes = training_x_PIV_data.shape[0] # 12 boxes

PIV_x_height = training_x_PIV_data.shape[2]
PIV_x_width = training_x_PIV_data.shape[3]

PLIF_height = training_PLIF_data.shape[2]
PLIF_width = training_PLIF_data.shape[3]

# then reshape datasets (i.e. flatten the 12 boxes)
training_x_PIV_data = training_x_PIV_data.reshape((boxes * training_nums, PIV_x_height, PIV_x_width))
training_PLIF_data = training_PLIF_data.reshape((boxes * training_nums, PLIF_height, PLIF_width))

validation_x_PIV_data = validation_x_PIV_data.reshape((boxes * validation_nums, PIV_x_height, PIV_x_width))
validation_PLIF_data = validation_PLIF_data.reshape((boxes * validation_nums, PLIF_height, PLIF_width))

testing_x_PIV_data = testing_x_PIV_data.reshape((boxes * testing_nums, PIV_x_height, PIV_x_width))
testing_PLIF_data = testing_PLIF_data.reshape((boxes * testing_nums, PLIF_height, PLIF_width))

# 3.6. obtain the training, validation, training sets
training_x_PIV_data = np.expand_dims(training_x_PIV_data, axis=1)
training_PLIF_data = np.expand_dims(training_PLIF_data, axis=1)

validation_x_PIV_data = np.expand_dims(validation_x_PIV_data, axis=1)
validation_PLIF_data = np.expand_dims(validation_PLIF_data, axis=1)

testing_x_PIV_data = np.expand_dims(testing_x_PIV_data, axis=1)
testing_PLIF_data = np.expand_dims(testing_PLIF_data, axis=1)

# 4.5. create the corresponding datasets
training_x_PIV_dataset = MyDataset(training_x_PIV_data)
training_PLIF_dataset = MyDataset(training_PLIF_data)

validation_x_PIV_dataset = MyDataset(validation_x_PIV_data)
validation_PLIF_dataset = MyDataset(validation_PLIF_data)

testing_x_PIV_dataset = MyDataset(testing_x_PIV_data)
testing_PLIF_dataset = MyDataset(testing_PLIF_data)

# 4.6. create the corresponding dataloaders
training_x_PIV_loader = DataLoader(dataset=training_x_PIV_dataset, batch_size=batch_size, shuffle=False)
training_PLIF_loader = DataLoader(dataset=training_PLIF_dataset, batch_size=batch_size, shuffle=False)

validation_x_PIV_loader = DataLoader(dataset=validation_x_PIV_dataset, batch_size=batch_size, shuffle=False)
validation_PLIF_loader = DataLoader(dataset=validation_PLIF_dataset, batch_size=batch_size, shuffle=False)

testing_x_PIV_loader = DataLoader(dataset=testing_x_PIV_dataset, batch_size=batch_size, shuffle=False)
testing_PLIF_loader = DataLoader(dataset=testing_PLIF_dataset, batch_size=batch_size, shuffle=False)

# PART 3: preparation before training the model
# 1. define the FullyCNN model
fullyCNN = FullyCNN()

# check if there is an existing model
if os.path.exists('./model/fullyCNN.pt'):
    fullyCNN = torch.load('./model/fullyCNN.pt')

    # set the if_existing flag
    if_existing = True
    print("Load the existing fullyCNN model, use this model to make prediction.")
else:
    print("Error: No existing fullyCNN model.")
    exit()

# 2. define the loss function
loss_fn = nn.MSELoss()

# PART 4: show the results via the trained model
# 1. calculate the loss for training, validation and testing datasets
training_loss = validate_epoch(fullyCNN=fullyCNN, device=device, dataloader_in=training_PLIF_loader,
                               dataloader_out=training_x_PIV_loader, loss_fn=loss_fn)
print(f"The MSE loss for the training dataset is {training_loss}.")

validation_loss = validate_epoch(fullyCNN=fullyCNN, device=device, dataloader_in=validation_PLIF_loader,
                                 dataloader_out=validation_x_PIV_loader, loss_fn=loss_fn)
print(f"The MSE loss for the validation dataset is {validation_loss}.")

test_loss = validate_epoch(fullyCNN=fullyCNN, device=device, dataloader_in=testing_PLIF_loader,
                           dataloader_out=testing_x_PIV_loader, loss_fn=loss_fn)
print(f"The MSE loss for the testing dataset is {test_loss}.")

# 2. show the difference and comparison of the results of training data
training_prediction = fullyCNN_predict(fullyCNN=fullyCNN, device=device, dataloader_in=training_PLIF_loader)
training_prediction = training_prediction.cpu().data.numpy()
training_prediction = training_prediction.reshape((boxes, training_nums, PIV_x_height, PIV_x_width))
training_x_PIV_data = training_x_PIV_data.reshape((boxes, training_nums, PIV_x_height, PIV_x_width))
training_difference = training_prediction - training_x_PIV_data

show_comparison(prediction_data=training_prediction, actual_data=training_x_PIV_data,
                prediction_filename="training_prediction.png", actual_filename="training_actual.png",
                rows=rows, columns=columns, img_num=img_num)

show_difference(difference=training_difference, filename="training_difference.png",
                rows=rows, columns=columns, img_num=img_num)

# 3. show the difference and comparison of the results of validation data
validation_prediction = fullyCNN_predict(fullyCNN=fullyCNN, device=device, dataloader_in=validation_PLIF_loader)
validation_prediction = validation_prediction.cpu().data.numpy()
validation_prediction = validation_prediction.reshape((boxes, validation_nums, PIV_x_height, PIV_x_width))
validation_x_PIV_data = validation_x_PIV_data.reshape((boxes, validation_nums, PIV_x_height, PIV_x_width))
validation_difference = validation_prediction - validation_x_PIV_data

show_comparison(prediction_data=validation_prediction, actual_data=validation_x_PIV_data,
                prediction_filename="validation_prediction.png", actual_filename="validation_actual.png",
                rows=rows, columns=columns, img_num=img_num)

show_difference(difference=validation_difference, filename="validataion_difference.png",
                rows=rows, columns=columns, img_num=img_num)

# 4. show the difference and comparison of the results of testing data
testing_prediction = fullyCNN_predict(fullyCNN=fullyCNN, device=device, dataloader_in=testing_PLIF_loader)
testing_prediction = testing_prediction.cpu().data.numpy()
testing_prediction = testing_prediction.reshape((boxes, testing_nums, PIV_x_height, PIV_x_width))
testing_x_PIV_data = testing_x_PIV_data.reshape((boxes, testing_nums, PIV_x_height, PIV_x_width))
testing_difference = testing_prediction - testing_x_PIV_data

show_comparison(prediction_data=testing_prediction, actual_data=testing_x_PIV_data,
                prediction_filename="testing_prediction.png", actual_filename="testing_actual.png",
                rows=rows, columns=columns, img_num=img_num)

show_difference(difference=testing_difference, filename="testing_difference.png",
                rows=rows, columns=columns, img_num=img_num)
