import torch
from torch import nn


def fullyCNN_predict(fullyCNN: nn.Module, device: torch.device, dataloader_in):

    # set evaluation mode for fullyCNN model
    fullyCNN.eval()

    with torch.no_grad():

        predicted_output = []

        for image_batch_in in dataloader_in:

            # move the tensor to device
            image_batch_in = image_batch_in.to(device)

            # pass the input data to the fullyCNN model
            predicted_data = fullyCNN(image_batch_in)

            predicted_output.append(predicted_data)

        return predicted_output
