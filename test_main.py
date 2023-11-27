import os.path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import summary

from globalCNN.train import train_epoch
from globalCNN.validate import validate_epoch
from preprocess_dataset import preprocess_data, preprocess_old_data, show_image, MyDataset, \
    show_box_PIV, show_box_PLIF, crop_old_data
from globalCNN.neural_net import FullyCNN
from result_visualiser import show_loss

"""
This file is used for testing the whole process of training the model.
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

# 3. reshape the datasets to 3 dimensions (i.e. flatten the 12 boxes)
boxes = PIV_x_attached_data.shape[0]
image_num = PIV_x_attached_data.shape[1]

PIV_x_height = PIV_x_attached_data.shape[2]
PIV_x_width = PIV_x_attached_data.shape[3]
PIV_x_attached_data = PIV_x_attached_data.reshape((boxes * image_num, PIV_x_height, PIV_x_width))

PLIF_height = PLIF_attached_data.shape[2]
PLIF_width = PLIF_attached_data.shape[3]
PLIF_attached_data = PLIF_attached_data.reshape((boxes * image_num, PLIF_height, PLIF_width))

# print(np.shape(PIV_x_attached_data)) # (24000, 14, 11)
# print(np.shape(PLIF_attached_data)) # (24000, 124, 93)

# 4. split the datasets for training, evaluation and testing
# 4.1. get the total number of data
attached_num = PIV_x_attached_data.shape[0]

# 4.2. calculate the position of splitting points to satisfy the proportion (3: 1: 1)
attached_split_points = [int(np.floor(attached_num * 0.6)), int(np.floor(attached_num * 0.8))]

# 4.3. split the datasets according to the splitting points
PIV_x_attached_data_split = np.split(PIV_x_attached_data, attached_split_points, axis=0)
PLIF_attached_data_split = np.split(PLIF_attached_data, attached_split_points, axis=0)

# 4.4. obtain the training, validation, training sets
training_x_PIV_data = np.expand_dims(PIV_x_attached_data_split[0], axis=1)
training_PLIF_data = np.expand_dims(PLIF_attached_data_split[0], axis=1)

validation_x_PIV_data = np.expand_dims(PIV_x_attached_data_split[1], axis=1)
validation_PLIF_data = np.expand_dims(PLIF_attached_data_split[1], axis=1)

testing_x_PIV_data = np.expand_dims(PIV_x_attached_data_split[2], axis=1)
testing_PLIF_data = np.expand_dims(PLIF_attached_data_split[2], axis=1)

# print(np.shape(training_x_PIV_data))
# print(np.shape(training_PLIF_data))
#
# print(np.shape(validation_x_PIV_data))
# print(np.shape(validation_PLIF_data))
#
# print(np.shape(testing_x_PIV_data))
# print(np.shape(testing_PLIF_data))

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
