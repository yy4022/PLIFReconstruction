import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import summary

from fullyCNN.neural_net import FullyCNN
from fullyCNN.train import train_epoch
from fullyCNN.validate import validate_epoch
from preprocess_methods import crop_old_PLIFdata, crop_old_PIVdata, get_min_max, preprocess_data_list, concatenate_data, \
    MyDataset
from result_visualiser import show_loss

"""
This file is used for testing the whole process of training the local model.
NOTE: 1. There is no need to use the testing datasets during the training process.
      2. The main difference between the global and channel method is how to use the datasets.
"""

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
EPOCHS = 1000
lr = 0.0001
if_existing = False # a flag recording if there is an existing fullyCNN model
specified_box = 1 # indicates which specified box corresponds to the local fullyCNN model, range in [1, 12]

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

# 2.2 shuffle the datasets by the index
# create a shuffled index array
index_array = np.arange(attached_num)
np.random.shuffle(index_array)

# shuffle the datasets according to the same shuffled index array
PLIF_attached_data = np.take(PLIF_attached_data, index_array, axis=1)
PIV_x_attached_data = np.take(PIV_x_attached_data, index_array, axis=1)
PIV_y_attached_data = np.take(PIV_y_attached_data, index_array, axis=1)
PIV_z_attached_data = np.take(PIV_z_attached_data, index_array, axis=1)

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

# 2.5. select the specified box to train
training_x_PIV_data = training_x_PIV_data[specified_box-1, :, :, :]
training_PLIF_data = training_PLIF_data[specified_box-1, :, :, :]

validation_x_PIV_data = validation_x_PIV_data[specified_box-1, :, :, :]
validation_PLIF_data = validation_PLIF_data[specified_box-1, :, :, :]

# 2.6. obtain the training, validation sets
training_x_PIV_data = np.expand_dims(training_x_PIV_data, axis=1)
training_PLIF_data = np.expand_dims(training_PLIF_data, axis=1)

validation_x_PIV_data = np.expand_dims(validation_x_PIV_data, axis=1)
validation_PLIF_data = np.expand_dims(validation_PLIF_data, axis=1)

# 2.7. create the corresponding datasets
training_x_PIV_dataset = MyDataset(training_x_PIV_data)
training_PLIF_dataset = MyDataset(training_PLIF_data)

validation_x_PIV_dataset = MyDataset(validation_x_PIV_data)
validation_PLIF_dataset = MyDataset(validation_PLIF_data)

# 2.8. create the corresponding dataloaders
training_x_PIV_loader = DataLoader(dataset=training_x_PIV_dataset, batch_size=batch_size, shuffle=False)
training_PLIF_loader = DataLoader(dataset=training_PLIF_dataset, batch_size=batch_size, shuffle=False)

validation_x_PIV_loader = DataLoader(dataset=validation_x_PIV_dataset, batch_size=batch_size, shuffle=False)
validation_PLIF_loader = DataLoader(dataset=validation_PLIF_dataset, batch_size=batch_size, shuffle=False)

# PART 3: preparation before training the model
# 1. define the FullyCNN model
fullyCNN = FullyCNN()

# check if there is an existing model
if os.path.exists(f'./model/fullyCNN_box{specified_box}.pt'):
    fullyCNN = torch.load('./model/fullyCNN.pt')

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
