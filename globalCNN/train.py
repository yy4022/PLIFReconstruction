import numpy as np
import torch
from torch import nn


def train_epoch(fullyCNN: nn.Module, device: torch.device, dataloader_in, dataloader_out,
                loss_fn, optimizer):

    # set the train mode for fullyCNN
    fullyCNN.train()

    train_epoch_loss = []

    for image_batch_in, image_batch_out in zip(dataloader_in, dataloader_out):

        # move tensor to the proper device
        image_batch_in = image_batch_in.to(device)
        image_batch_out = image_batch_out.to(device)

        # pass the input images to the fullyCNN model
        predicted_data = fullyCNN(image_batch_in)

        # compute the prediction loss
        loss = loss_fn(predicted_data, image_batch_out)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_epoch_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_epoch_loss)
