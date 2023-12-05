import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import summary

from fullyCNN.neural_net import FullyCNN
from fullyCNN.train import train_epoch
from fullyCNN.validate import validate_epoch
from preprocess_methods import MyDataset
from result_visualiser import show_loss

"""
This file is used for testing the whole process of training the local model.
NOTE: there is no need to use the testing datasets during the training process.
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
img_num = 0
EPOCHS = 9000
lr = 0.0001
if_existing = False  # a flag recording if there is an existing fullyCNN model
dataset_num = 4
specified_box = 2  # indicates which specified box corresponds to the local fullyCNN model, range in [1, 12]

# PART 2：create the dataloader for training the model of the specified box
# 2.1. read the specified box image data from all data files
training_PLIF_data_box = []
training_x_PIV_data_box = []

validation_PLIF_data_box = []
validation_x_PIV_data_box = []

# use loop to collect all data of the specified box
for i in range(dataset_num):

    training_PLIF_data = np.load(f'data/Preprocessed_Data_old/training_PLIF_data{i + 1}.npy')
    training_x_PIV_data = np.load(f'data/Preprocessed_Data_old/training_x_PIV_data{i + 1}.npy')
    training_PLIF_data_box.append(training_PLIF_data[specified_box - 1, :, :, :])
    training_x_PIV_data_box.append(training_x_PIV_data[specified_box - 1, :, :, :])

    validation_PLIF_data = np.load(f'data/Preprocessed_Data_old/validation_PLIF_data{i + 1}.npy')
    validation_x_PIV_data = np.load(f'data/Preprocessed_Data_old/validation_x_PIV_data{i + 1}.npy')
    validation_PLIF_data_box.append(validation_PLIF_data[specified_box - 1, :, :, :])
    validation_x_PIV_data_box.append(validation_x_PIV_data[specified_box - 1, :, :, :])

training_PLIF_data_box = np.concatenate(training_PLIF_data_box, axis=0)
training_x_PIV_data_box = np.concatenate(training_x_PIV_data_box, axis=0)

validation_PLIF_data_box = np.concatenate(validation_PLIF_data_box, axis=0)
validation_x_PIV_data_box = np.concatenate(validation_x_PIV_data_box, axis=0)

# obtain the training, validation sets
training_x_PIV_data = np.expand_dims(training_x_PIV_data_box, axis=1)
training_PLIF_data = np.expand_dims(training_PLIF_data_box, axis=1)

validation_x_PIV_data = np.expand_dims(validation_x_PIV_data_box, axis=1)
validation_PLIF_data = np.expand_dims(validation_PLIF_data_box, axis=1)

# 2.2. create the corresponding datasets
training_PLIF_dataset = MyDataset(training_PLIF_data)
training_x_PIV_dataset = MyDataset(training_x_PIV_data)

validation_PLIF_dataset = MyDataset(validation_PLIF_data)
validation_x_PIV_dataset = MyDataset(validation_x_PIV_data)

# 2.3. create the corresponding dataloaders
training_x_PIV_loader = DataLoader(dataset=training_x_PIV_dataset, batch_size=batch_size, shuffle=False)
training_PLIF_loader = DataLoader(dataset=training_PLIF_dataset, batch_size=batch_size, shuffle=False)

validation_x_PIV_loader = DataLoader(dataset=validation_x_PIV_dataset, batch_size=batch_size, shuffle=False)
validation_PLIF_loader = DataLoader(dataset=validation_PLIF_dataset, batch_size=batch_size, shuffle=False)

# PART 3: preparation before training the model
# 1. define the FullyCNN model
fullyCNN = FullyCNN()

# check if there is an existing model
if os.path.exists(f'./model/fullyCNN_box{specified_box}.pt'):
    fullyCNN = torch.load(f'./model/fullyCNN_box{specified_box}.pt')

    # set the if_existing flag
    if_existing = True
    print(f"Load the existing fullyCNN model of box{specified_box}, then continue training.")
else:
    print(f"No existing fullyCNN model of box{specified_box}, so create a new one.")

fullyCNN = fullyCNN.to(device)
input_shape = (1, 124, 93)
summary(fullyCNN, input_shape)

# 2. create a numpy array for recording the loss,
#   and set the best (validation) loss for updating the model
train_loss_records = []
validation_loss_records = []

train_loss_records = np.array(train_loss_records)
validation_loss_records = np.array(validation_loss_records)

best_loss = 10.0

if if_existing == True:
    train_loss_records = \
        np.append(train_loss_records, np.load(f'./result/train_loss_records_box{specified_box}.npy'))

    validation_loss_records = \
        np.append(validation_loss_records, np.load(f'./result/validation_loss_records_box{specified_box}.npy'))

    best_loss = validation_loss_records.min()
    print(f"Load the existing loss records, and current best loss is {best_loss}.")

else:
    print("No existing loss records, start recording from the beginning.")

# 3. define the loss function and the optimizer
loss_fn = nn.MSELoss()

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

optimizer = torch.optim.Adam(fullyCNN.parameters(), lr=lr, weight_decay=1e-05)

# PART 4: the looping process of training the model
# NOTE: the test file takes PIV-x (dimension-0) as an example
for epoch in range(EPOCHS):

    train_loss = train_epoch(fullyCNN=fullyCNN, device=device, dataloader_in=training_PLIF_loader,
                             dataloader_out=training_x_PIV_loader, loss_fn=loss_fn, optimizer=optimizer)

    validation_loss = validate_epoch(fullyCNN=fullyCNN, device=device, dataloader_in=validation_PLIF_loader,
                                     dataloader_out=validation_x_PIV_loader, loss_fn=loss_fn)

    print(
        '\n EPOCH {}/{} \t train loss {} \t validate loss {}'.format(epoch + 1, EPOCHS, train_loss,
                                                                     validation_loss))

    train_loss_records = np.append(train_loss_records, train_loss)
    validation_loss_records = np.append(validation_loss_records, validation_loss)

    if validation_loss < best_loss:
        best_loss = validation_loss
        torch.save(fullyCNN, f'./model/fullyCNN_box{specified_box}.pt')

# save loss records of training and validation process
np.save(f"./result/train_loss_records_box{specified_box}.npy", train_loss_records)
np.save(f"./result/validation_loss_records_box{specified_box}.npy", validation_loss_records)

loss_records = {
    'train_loss_records': train_loss_records,
    'validation_loss_records': validation_loss_records
}

# PART 5: show the results
# 5.1. show the loss records of the whole training process
show_loss(loss_records, f"fullyCNN_loss_box{specified_box}.png")

