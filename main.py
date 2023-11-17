import h5py
import numpy as np
import scipy.io
import matplotlib as mpl
from matplotlib import pyplot as plt

from preprocess_dataset import preprocess_data, preprocess_old_data, show_image

# PART 1: define the parameters
file_PIV = str('data/Lifted state/D1F1_air240_PIV_13601to14000.mat')
file_PLIF = str('data/Lifted state/D1F1_air240_PLIF_13601to14000.mat')
# file_PIV = str('data/IA_PIV.mat')
# file_PLIF = str('data/IA_PLIF_1to2500.mat')

# PART 2: preprocess the datasets
# 1. concatenate the datasets as required

# 2. preprocess the data, then return the cropped datasets
PIV_data, PLIF_data, xmin, xmax, ymin, ymax = preprocess_old_data(file_PIV, file_PLIF)

# test part
show_image(PIV_data[:, 1, :, :], PLIF_data[1, :, :],
           xmin, xmax, ymin, ymax)

