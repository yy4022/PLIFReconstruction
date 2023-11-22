from torch import nn


# Build a fully CNN class
class FullyCNN(nn.Module):
    def __init__(self):

        super().__init__()

        # define the encoder of the fully CNN structure
        self.encoder_cnn = nn.Sequential()

        # define the decoder of the fully CNN structure
        self.decoder_cnn = nn.Sequential()

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.decoder_cnn(x)
        return x
