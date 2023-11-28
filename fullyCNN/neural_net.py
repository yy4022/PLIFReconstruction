import torch
from torch import nn


# Build a fully CNN class
class FullyCNN(nn.Module):
    def __init__(self):

        super().__init__()

        # define the encoder of the fully CNN structure
        # NOTE: downsampling phase -> doubly strided conv, instead of max-pooling
        self.encoder_cnn = nn.Sequential(
            # 1st Convolutional 2D layer
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=8, stride=2),
            # ReLU activation function
            nn.ReLU(),

            # 2nd Convolutional 2D layer
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=4, stride=2),
            # ReLU activation function
            nn.ReLU(),

            # 3rd Convolutional 2D layer
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=4, stride=2),
            # ReLU activation function
            nn.ReLU(),

            # 4th Convolutional 2D layer
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=2),
        )

        # define the decoder of the fully CNN structure
        # NOTE: upsampling phase -> transpose conv, instead of un-pooling
        self.decoder_cnn = nn.Sequential(
            # 1st Transpose Convolutional layer
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=4, stride=2),
            # ReLU activation function
            nn.ReLU(),

            # 2nd Transpose Convolutional layer
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(3, 4), stride=1)
        )

    def forward(self, x):
        # x is a 2D image
        x = self.encoder_cnn(x)
        x = self.decoder_cnn(x)
        return x
