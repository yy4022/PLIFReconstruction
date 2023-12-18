import os.path
import pickle

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from fullyCNN.neural_net import FullyCNN
from fullyCNN.predict import fullyCNN_predict
from fullyCNN.validate import validate_epoch
from methods_show import show_difference, show_comparison

"""
This file is used for showing the results via the trained global model.
"""

# PART 1: define the parameters
# 1.1. choose the device
torch.cuda.set_device(0)

print(torch.__version__)
print(torch.cuda.is_available())
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Selected device: {device}")

# 1.2. define the parameters for training the model
batch_size = 200
rows = 3
columns = 4
boxes = rows * columns
img_num = 0
specified_dataset = 1  # range in [1, 4]

# PART 2: load the existing model for showing results
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

# PART 3. show the training set results
# 3.1. load the numpy array
training_x_PIV_data = np.load(f'data/Preprocessed_Data_old/training_x_PIV_data{specified_dataset}.npy')
PIV_x_height = training_x_PIV_data.shape[2]
PIV_x_width = training_x_PIV_data.shape[3]

# 3.2. load the dataset
with open(f'data/Preprocessed_Data_old/training_PLIF_dataset{specified_dataset}.pkl', 'rb') as file:
    training_PLIF_dataset = pickle.load(file)

with open(f'data/Preprocessed_Data_old/training_x_PIV_dataset{specified_dataset}.pkl', 'rb') as file:
    training_x_PIV_dataset = pickle.load(file)

# 3.3. create the dataloader
training_PLIF_loader = DataLoader(dataset=training_PLIF_dataset, batch_size=batch_size, shuffle=False)
training_x_PIV_loader = DataLoader(dataset=training_x_PIV_dataset, batch_size=batch_size, shuffle=False)

# 3.4. calculate the loss for training dataset
training_loss = validate_epoch(fullyCNN=fullyCNN, device=device, dataloader_in=training_PLIF_loader,
                               dataloader_out=training_x_PIV_loader, loss_fn=loss_fn)
print(f"The MSE loss for the training dataset is {training_loss}.")

# 3.5. show the difference and comparison of the results of training data
training_prediction = fullyCNN_predict(fullyCNN=fullyCNN, device=device, dataloader_in=training_PLIF_loader)
training_prediction = training_prediction.cpu().data.numpy()
training_prediction = training_prediction.reshape((boxes, -1, PIV_x_height, PIV_x_width))
training_x_PIV_data = training_x_PIV_data.reshape((boxes, -1, PIV_x_height, PIV_x_width))
training_difference = training_prediction - training_x_PIV_data

show_comparison(prediction_data=training_prediction, actual_data=training_x_PIV_data,
                prediction_filename="training_prediction.png", actual_filename="training_actual.png",
                rows=rows, columns=columns, img_num=img_num)

show_difference(difference=training_difference, filename="training_difference.png",
                rows=rows, columns=columns, img_num=img_num)

print(np.amin(training_prediction))
print(np.amax(training_prediction))
print()
print(np.amin(training_x_PIV_data))
print(np.amax(training_x_PIV_data))

# PART 4. show the validation set results
# 4.1. load the numpy array
validation_x_PIV_data = np.load(f'data/Preprocessed_Data_old/validation_x_PIV_data{specified_dataset}.npy')
PIV_x_height = validation_x_PIV_data.shape[2]
PIV_x_width = validation_x_PIV_data.shape[3]

# 4.2. load the dataset
with open(f'data/Preprocessed_Data_old/validation_PLIF_dataset{specified_dataset}.pkl', 'rb') as file:
    validation_PLIF_dataset = pickle.load(file)

with open(f'data/Preprocessed_Data_old/validation_x_PIV_dataset{specified_dataset}.pkl', 'rb') as file:
    validation_x_PIV_dataset = pickle.load(file)

# 4.3. create the dataloader
validation_x_PIV_loader = DataLoader(dataset=validation_x_PIV_dataset, batch_size=batch_size, shuffle=False)
validation_PLIF_loader = DataLoader(dataset=validation_PLIF_dataset, batch_size=batch_size, shuffle=False)

# 4.4. calculate the loss for training dataset
validation_loss = validate_epoch(fullyCNN=fullyCNN, device=device, dataloader_in=validation_PLIF_loader,
                                 dataloader_out=validation_x_PIV_loader, loss_fn=loss_fn)
print(f"The MSE loss for the validation dataset is {validation_loss}.")

# 4.5. show the difference and comparison of the results of training data
validation_prediction = fullyCNN_predict(fullyCNN=fullyCNN, device=device, dataloader_in=validation_PLIF_loader)
validation_prediction = validation_prediction.cpu().data.numpy()
validation_prediction = validation_prediction.reshape((boxes, -1, PIV_x_height, PIV_x_width))
validation_x_PIV_data = validation_x_PIV_data.reshape((boxes, -1, PIV_x_height, PIV_x_width))
validation_difference = validation_prediction - validation_x_PIV_data

show_comparison(prediction_data=validation_prediction, actual_data=validation_x_PIV_data,
                prediction_filename="validation_prediction.png", actual_filename="validation_actual.png",
                rows=rows, columns=columns, img_num=img_num)

show_difference(difference=validation_difference, filename="validataion_difference.png",
                rows=rows, columns=columns, img_num=img_num)

# PART 5. show the validation set results
# 5.1. load the numpy array
testing_x_PIV_data = np.load(f'data/Preprocessed_Data_old/testing_x_PIV_data{specified_dataset}.npy')
PIV_x_height = testing_x_PIV_data.shape[2]
PIV_x_width = testing_x_PIV_data.shape[3]

# 5.2. load the dataset
with open(f'data/Preprocessed_Data_old/testing_PLIF_dataset{specified_dataset}.pkl', 'rb') as file:
    testing_PLIF_dataset = pickle.load(file)

with open(f'data/Preprocessed_Data_old/testing_x_PIV_dataset{specified_dataset}.pkl', 'rb') as file:
    testing_x_PIV_dataset = pickle.load(file)

# 5.3. create the dataloader
testing_x_PIV_loader = DataLoader(dataset=testing_x_PIV_dataset, batch_size=batch_size, shuffle=False)
testing_PLIF_loader = DataLoader(dataset=testing_PLIF_dataset, batch_size=batch_size, shuffle=False)

# 5.4. calculate the loss for training dataset
test_loss = validate_epoch(fullyCNN=fullyCNN, device=device, dataloader_in=testing_PLIF_loader,
                           dataloader_out=testing_x_PIV_loader, loss_fn=loss_fn)
print(f"The MSE loss for the testing dataset is {test_loss}.")

# 5.5. show the difference and comparison of the results of training data
testing_prediction = fullyCNN_predict(fullyCNN=fullyCNN, device=device, dataloader_in=testing_PLIF_loader)
testing_prediction = testing_prediction.cpu().data.numpy()
testing_prediction = testing_prediction.reshape((boxes, -1, PIV_x_height, PIV_x_width))
testing_x_PIV_data = testing_x_PIV_data.reshape((boxes, -1, PIV_x_height, PIV_x_width))
testing_difference = testing_prediction - testing_x_PIV_data

show_comparison(prediction_data=testing_prediction, actual_data=testing_x_PIV_data,
                prediction_filename="testing_prediction.png", actual_filename="testing_actual.png",
                rows=rows, columns=columns, img_num=img_num)

show_difference(difference=testing_difference, filename="testing_difference.png",
                rows=rows, columns=columns, img_num=img_num)
