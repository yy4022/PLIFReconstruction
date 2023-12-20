import os.path
import pickle

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import summary

from fullyCNN.neural_net import FullyCNN
from fullyCNN.train import train_epoch
from fullyCNN.validate import validate_epoch
from methods_show import show_loss

"""
This file is used for testing the whole process of training the global model.
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
EPOCHS = 1000
lr = 0.0001
if_existing = False  # a flag recording if there is an existing fullyCNN model
dataset_num = 7

# PART 2：create the dataloader for training
# 2.1. load the datasets
training_PLIF_dataset_list = []
training_x_PIV_dataset_list = []

validation_PLIF_dataset_list = []
validation_x_PIV_dataset_list = []

for i in range(dataset_num):
    with open(f'data/Preprocessed_Data_Fulldataset/data_PLIF/training_PLIF_dataset{i + 1}.pkl', 'rb') as file:
        training_PLIF_dataset = pickle.load(file)

    with open(f'data/Preprocessed_Data_Fulldataset/data_PIV/training_PIV_x_dataset{i + 1}.pkl', 'rb') as file:
        training_x_PIV_dataset = pickle.load(file)

    with open(f'data/Preprocessed_Data_Fulldataset/data_PLIF/validation_PLIF_dataset{i + 1}.pkl', 'rb') as file:
        validation_PLIF_dataset = pickle.load(file)

    with open(f'data/Preprocessed_Data_Fulldataset/data_PIV/validation_PIV_x_dataset{i + 1}.pkl', 'rb') as file:
        validation_x_PIV_dataset = pickle.load(file)

    training_PLIF_dataset_list.append(training_PLIF_dataset)
    training_x_PIV_dataset_list.append(training_x_PIV_dataset)
    validation_PLIF_dataset_list.append(validation_PLIF_dataset)
    validation_x_PIV_dataset_list.append(validation_x_PIV_dataset)

# 2.2. create the corresponding dataloaders
training_PLIF_dataloader_list = []
training_x_PIV_dataloader_list = []

validation_PLIF_dataloader_list = []
validation_x_PIV_dataloader_list = []

for i in range(dataset_num):
    training_x_PIV_loader = DataLoader(dataset=training_x_PIV_dataset_list[i], batch_size=batch_size, shuffle=False)
    training_PLIF_loader = DataLoader(dataset=training_PLIF_dataset_list[i], batch_size=batch_size, shuffle=False)

    validation_x_PIV_loader = DataLoader(dataset=validation_x_PIV_dataset_list[i], batch_size=batch_size, shuffle=False)
    validation_PLIF_loader = DataLoader(dataset=validation_PLIF_dataset_list[i], batch_size=batch_size, shuffle=False)

    training_PLIF_dataloader_list.append(training_PLIF_loader)
    training_x_PIV_dataloader_list.append(training_x_PIV_loader)
    validation_PLIF_dataloader_list.append(validation_PLIF_loader)
    validation_x_PIV_dataloader_list.append(validation_x_PIV_loader)

# PART 3: preparation before training the model
# 1. define the FullyCNN model
fullyCNN = FullyCNN()

# check if there is an existing model
if os.path.exists('./model/fullyCNN.pt'):
    fullyCNN = torch.load('./model/fullyCNN.pt')

    # set if_existing flag
    if_existing = True
    print("Load the existing fullyCNN model, then continue training.")
else:
    print("No existing fullyCNN model, so create a new one.")

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
        np.append(train_loss_records, np.load('./result/train_loss_records.npy'))

    validation_loss_records = \
        np.append(validation_loss_records, np.load('./result/validation_loss_records.npy'))

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

    train_loss_list = []
    validation_loss_list = []

    for i in range(dataset_num):
        train_loss_i = train_epoch(fullyCNN=fullyCNN, device=device, dataloader_in=training_PLIF_dataloader_list[i],
                                   dataloader_out=training_x_PIV_dataloader_list[i],
                                   loss_fn=loss_fn, optimizer=optimizer)

        validation_loss_i = validate_epoch(fullyCNN=fullyCNN, device=device,
                                           dataloader_in=validation_PLIF_dataloader_list[i],
                                           dataloader_out=validation_x_PIV_dataloader_list[i], loss_fn=loss_fn)

        train_loss_list.append(train_loss_i)
        validation_loss_list.append(validation_loss_i)

    train_loss = np.mean(train_loss_list)
    validation_loss = np.mean(validation_loss_list)

    print(
        '\n EPOCH {}/{} \t train loss {} \t validate loss {}'.format(epoch + 1, EPOCHS, train_loss,
                                                                     validation_loss))

    train_loss_records = np.append(train_loss_records, train_loss)
    validation_loss_records = np.append(validation_loss_records, validation_loss)

    if validation_loss < best_loss:
        best_loss = validation_loss
        torch.save(fullyCNN, './model/fullyCNN.pt')

# save loss records of training and validation process
np.save("./result/train_loss_records.npy", train_loss_records)
np.save("./result/validation_loss_records.npy", validation_loss_records)

loss_records = {
    'train_loss_records': train_loss_records,
    'validation_loss_records': validation_loss_records
}

# PART 5: show the results
# 5.1. show the loss records of the whole training process
show_loss(loss_records, "fullyCNN_loss.png")
