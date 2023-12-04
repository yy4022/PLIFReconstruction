

"""
This file is used for showing the results via the trained local model.
NOTE: the main difference between local and global method is that how to call the model.
"""
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import summary

from fullyCNN.neural_net import FullyCNN
from fullyCNN.predict import fullyCNN_predict
from fullyCNN.validate import validate_epoch
from preprocess_methods import crop_old_PLIFdata, crop_old_PIVdata, get_min_max, preprocess_data_list, concatenate_data, \
    MyDataset
from result_visualiser import show_comparison, show_difference

# PART 1: define the parameters
# 1.1. choose the device
torch.cuda.set_device(0)

print(torch.__version__)
print(torch.cuda.is_available())
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Selected device: {device}")

# 1.2. define the parameters for training the model
batch_size = 100
rows = 3
columns = 4
img_num = 0

# 1.3. provide filenames of PIV, PLIF data
files_PIV = ['data/Attached state/D1F1_air240_PIV_1001to2000.mat',
             'data/Attached state/D1F1_air240_PIV_2001to3000.mat']

files_PLIF = ['data/Attached state/D1F1_air240_PLIF_1001to2000.mat',
              'data/Attached state/D1F1_air240_PLIF_2001to3000.mat']

# PART 2: preprocess the datasets
# 2.1. preprocess the datasets, then return the cropped datasets
cropped_PLIF_data = crop_old_PLIFdata(files_PLIF)
cropped_PIV_x_data, cropped_PIV_y_data, cropped_PIV_z_data = crop_old_PIVdata(files_PIV)

# 2.2. get the min and max value for all PIV-x, y, z and PLIF datasets
min_PLIF, max_PLIF = get_min_max(cropped_PLIF_data)
min_PIV_x, max_PIV_x = get_min_max(cropped_PIV_x_data)
min_PIV_y, max_PIV_y = get_min_max(cropped_PIV_y_data)
min_PIV_z, max_PIV_z = get_min_max(cropped_PIV_z_data)

# 2.3. normalize and discretize the datasets according to the min, max values
PLIF_data = preprocess_data_list(cropped_PLIF_data, min_PLIF, max_PLIF)
PIV_x_data = preprocess_data_list(cropped_PIV_x_data, min_PIV_x, max_PIV_x)
PIV_y_data = preprocess_data_list(cropped_PIV_y_data, min_PIV_y, max_PIV_y)
PIV_z_data = preprocess_data_list(cropped_PIV_z_data, min_PIV_z, max_PIV_z)

# PART 3: split the datasets to training, validation, testing datasets
# 1. concatenate the datasets as required
PLIF_attached_data = concatenate_data(PLIF_data)
PIV_x_attached_data = concatenate_data(PIV_x_data)
PIV_y_attached_data = concatenate_data(PIV_y_data)
PIV_z_attached_data = concatenate_data(PIV_z_data)

# 2. split the datasets for training, validation and testing
# 2.1. get the total number of data
attached_num = PLIF_attached_data.shape[1]

# 2.2. calculate the position of splitting points to satisfy the proportion (3: 1: 1)
attached_split_points = [int(np.floor(attached_num * 0.6)), int(np.floor(attached_num * 0.8))]

# 2.3. split the datasets according to the splitting points
PIV_x_attached_data_split = np.split(PIV_x_attached_data, attached_split_points, axis=1)
PLIF_attached_data_split = np.split(PLIF_attached_data, attached_split_points, axis=1)

# 2.4. obtain the training, validation, testing sets
training_x_PIV_data = PIV_x_attached_data_split[0]
training_PLIF_data = PLIF_attached_data_split[0]
training_nums = training_PLIF_data.shape[1]

validation_x_PIV_data = PIV_x_attached_data_split[1]
validation_PLIF_data = PLIF_attached_data_split[1]
validation_nums = validation_PLIF_data.shape[1]

testing_x_PIV_data = PIV_x_attached_data_split[2]
testing_PLIF_data = PLIF_attached_data_split[2]
testing_nums = testing_PLIF_data.shape[1]

# PART 4: process the data, load the model of the specified box, then make predictions
training_prediction = []
validation_prediction = []
testing_prediction = []

training_prediction = np.array(training_prediction)
validation_prediction = np.array(validation_prediction)
testing_prediction = np.array(testing_prediction)

# use the loop to make prediction for all boxes
for specified_box in range(rows * columns):

    # Section 1: process the data of the specified box
    # 1.1. select the specified box data
    training_x_PIV_data_box = training_x_PIV_data[specified_box - 1, :, :, :]
    training_PLIF_data_box = training_PLIF_data[specified_box - 1, :, :, :]

    validation_x_PIV_data_box = validation_x_PIV_data[specified_box - 1, :, :, :]
    validation_PLIF_data_box = validation_PLIF_data[specified_box - 1, :, :, :]

    testing_x_PIV_data_box = testing_x_PIV_data[specified_box - 1, :, :, :]
    testing_PLIF_data_box = testing_PLIF_data[specified_box - 1, :, :, :]

    # 1.2. obtain the training, validation and testing datasets
    training_x_PIV_data_box = np.expand_dims(training_x_PIV_data_box, axis=1)
    training_PLIF_data_box = np.expand_dims(training_PLIF_data_box, axis=1)

    validation_x_PIV_data_box = np.expand_dims(validation_x_PIV_data_box, axis=1)
    validation_PLIF_data_box = np.expand_dims(validation_PLIF_data_box, axis=1)

    testing_x_PIV_data_box = np.expand_dims(testing_x_PIV_data_box, axis=1)
    testing_PLIF_data_box = np.expand_dims(testing_PLIF_data_box, axis=1)

    # 1.3. create the corresponding datasets
    training_x_PIV_dataset = MyDataset(training_x_PIV_data_box)
    training_PLIF_dataset = MyDataset(training_PLIF_data_box)

    validation_x_PIV_dataset = MyDataset(validation_x_PIV_data_box)
    validation_PLIF_dataset = MyDataset(validation_PLIF_data_box)

    testing_x_PIV_dataset = MyDataset(testing_x_PIV_data_box)
    testing_PLIF_dataset = MyDataset(testing_PLIF_data_box)

    # 1.4. create the corresponding dataloaders
    training_x_PIV_loader = DataLoader(dataset=training_x_PIV_dataset, batch_size=batch_size, shuffle=False)
    training_PLIF_loader = DataLoader(dataset=training_PLIF_dataset, batch_size=batch_size, shuffle=False)

    validation_x_PIV_loader = DataLoader(dataset=validation_x_PIV_dataset, batch_size=batch_size, shuffle=False)
    validation_PLIF_loader = DataLoader(dataset=validation_PLIF_dataset, batch_size=batch_size, shuffle=False)

    testing_x_PIV_loader = DataLoader(dataset=testing_x_PIV_dataset, batch_size=batch_size, shuffle=False)
    testing_PLIF_loader = DataLoader(dataset=testing_PLIF_dataset, batch_size=batch_size, shuffle=False)

    # Section 2: load the fullyCNN model of the specified box
    # 2.1. define the FullyCNN model
    fullyCNN = FullyCNN()

    # check if there is an existing model
    if os.path.exists(f'./model/fullyCNN_box{specified_box}.pt'):
        fullyCNN = torch.load('./model/fullyCNN.pt')

        # set the if_existing flag
        if_existing = True
        print(f"Load the existing fullyCNN model of box{specified_box}, use this model to make prediction.")
    else:
        print(f"Error: No existing fullyCNN model of box{specified_box}.")
        exit()

    # 2.2. define the loss function
    loss_fn = nn.MSELoss()

    # Section 3: show the loss via the trained model
    training_loss = validate_epoch(fullyCNN=fullyCNN, device=device, dataloader_in=training_PLIF_loader,
                                   dataloader_out=training_x_PIV_loader, loss_fn=loss_fn)
    print(f"The MSE loss of box{specified_box} for the training dataset is {training_loss}.")

    validation_loss = validate_epoch(fullyCNN=fullyCNN, device=device, dataloader_in=validation_PLIF_loader,
                                     dataloader_out=validation_x_PIV_loader, loss_fn=loss_fn)
    print(f"The MSE loss of box{specified_box} for the validation dataset is {validation_loss}.")

    test_loss = validate_epoch(fullyCNN=fullyCNN, device=device, dataloader_in=testing_PLIF_loader,
                               dataloader_out=testing_x_PIV_loader, loss_fn=loss_fn)
    print(f"The MSE loss of box{specified_box} for the testing dataset is {test_loss}.")
    print()

    # Section 4: use fullyCNN models to make predictions for specified box
    training_prediction_box = fullyCNN_predict(fullyCNN=fullyCNN, device=device, dataloader_in=training_PLIF_loader)
    training_prediction_box = training_prediction_box.cpu().data.numpy()

    validation_prediction_box = fullyCNN_predict(fullyCNN=fullyCNN, device=device, dataloader_in=validation_PLIF_loader)
    validation_prediction_box = validation_prediction_box.cpu().data.numpy()

    testing_prediction_box = fullyCNN_predict(fullyCNN=fullyCNN, device=device, dataloader_in=testing_PLIF_loader)
    testing_prediction_box = testing_prediction_box.cpu().data.numpy()

    # save the prediction of specified box to the result array
    training_prediction[specified_box, :, :, :] = training_prediction_box
    validation_prediction[specified_box, :, :, :] = validation_prediction_box
    testing_prediction[specified_box, :, :, :] = testing_prediction_box

# PART 5: show the comparison and difference between prediction and actual data
# 5.1. for the training dataset
show_comparison(prediction_data=training_prediction, actual_data=training_x_PIV_data,
                prediction_filename="training_prediction.png", actual_filename="training_actual.png",
                rows=rows, columns=columns, img_num=img_num)

training_difference = training_prediction - training_x_PIV_data
show_difference(difference=training_difference, filename="training_difference.png",
                rows=rows, columns=columns, img_num=img_num)

# 5.2. for the validation dataset
show_comparison(prediction_data=validation_prediction, actual_data=validation_x_PIV_data,
                prediction_filename="validation_prediction.png", actual_filename="validation_actual.png",
                rows=rows, columns=columns, img_num=img_num)

validation_difference = validation_prediction - validation_x_PIV_data
show_difference(difference=validation_difference, filename="validataion_difference.png",
                rows=rows, columns=columns, img_num=img_num)

# 5.3. for the testing dataset
show_comparison(prediction_data=testing_prediction, actual_data=testing_x_PIV_data,
                prediction_filename="testing_prediction.png", actual_filename="testing_actual.png",
                rows=rows, columns=columns, img_num=img_num)

testing_difference = testing_prediction - testing_x_PIV_data
show_difference(difference=testing_difference, filename="testing_difference.png",
                rows=rows, columns=columns, img_num=img_num)
