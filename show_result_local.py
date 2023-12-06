import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from fullyCNN.neural_net import FullyCNN
from fullyCNN.predict import fullyCNN_predict
from fullyCNN.validate import validate_epoch
from preprocess_methods import MyDataset
from show_methods import show_comparison, show_difference

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

# PART 2. show the results of the trained model
# 2.1. load the numpy array
training_PLIF_data = np.load(f'data/Preprocessed_Data_old/training_PLIF_data{specified_dataset}.npy')
validation_PLIF_data = np.load(f'data/Preprocessed_Data_old/validation_PLIF_data{specified_dataset}.npy')
testing_PLIF_data = np.load(f'data/Preprocessed_Data_old/testing_PLIF_data{specified_dataset}.npy')

training_x_PIV_data = np.load(f'data/Preprocessed_Data_old/training_x_PIV_data{specified_dataset}.npy')
validation_x_PIV_data = np.load(f'data/Preprocessed_Data_old/validation_x_PIV_data{specified_dataset}.npy')
testing_x_PIV_data = np.load(f'data/Preprocessed_Data_old/testing_x_PIV_data{specified_dataset}.npy')

# 2.2. process the specified dataset, load the model of the specified box, then make predictions
training_prediction = np.empty_like(training_x_PIV_data)
validation_prediction = np.empty_like(validation_x_PIV_data)
testing_prediction = np.empty_like(testing_x_PIV_data)

# use the loop to make prediction for all boxes
for i in range(rows * columns):

    specified_box = i + 1

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
        fullyCNN = torch.load(f'./model/fullyCNN_box{specified_box}.pt')

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
    training_prediction[specified_box - 1] = np.squeeze(training_prediction_box, axis=1)
    validation_prediction[specified_box - 1] = np.squeeze(validation_prediction_box, axis=1)
    testing_prediction[specified_box - 1] = np.squeeze(testing_prediction_box, axis=1)

# 2.3. show the comparison and difference between prediction and actual data
# 2.3.1. for the training dataset
show_comparison(prediction_data=training_prediction, actual_data=training_x_PIV_data,
                prediction_filename="training_prediction.png", actual_filename="training_actual.png",
                rows=rows, columns=columns, img_num=img_num)

training_difference = training_prediction - training_x_PIV_data
show_difference(difference=training_difference, filename="training_difference.png",
                rows=rows, columns=columns, img_num=img_num)

# 2.3.2. for the validation dataset
show_comparison(prediction_data=validation_prediction, actual_data=validation_x_PIV_data,
                prediction_filename="validation_prediction.png", actual_filename="validation_actual.png",
                rows=rows, columns=columns, img_num=img_num)

validation_difference = validation_prediction - validation_x_PIV_data
show_difference(difference=validation_difference, filename="validataion_difference.png",
                rows=rows, columns=columns, img_num=img_num)

# 2.3.3. for the testing dataset
show_comparison(prediction_data=testing_prediction, actual_data=testing_x_PIV_data,
                prediction_filename="testing_prediction.png", actual_filename="testing_actual.png",
                rows=rows, columns=columns, img_num=img_num)

testing_difference = testing_prediction - testing_x_PIV_data
show_difference(difference=testing_difference, filename="testing_difference.png",
                rows=rows, columns=columns, img_num=img_num)


