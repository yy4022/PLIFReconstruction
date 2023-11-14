import h5py
import numpy as np
import matplotlib as mpl

"""
This file is provided for visualising the datasets for training, validating and testing model
"""

# PART 1: define the parameters
first_PLIF_num = int(5001)
show_num = 6000
show_start = 6500
show_end = 6510

# PART 2: load the datasets
with h5py.File('data/IA_PIV.mat', 'r') as file:

    # get the PIV numpy array with the shape of (3, 5000, 73, 73)
    dataset_PIV = file['PIV']['velfield'][:]

    # get the x range of PIV image with (73, 1)
    PIV_x = file['PIV']['x'][:]

    # get the y range of PIV image with (73, 1)
    PIV_y = file['PIV']['y'][:]

with h5py.File('data/ID_PLIF_5001to6689.mat', 'r') as file:

    # get the PLIF numpy array with the shape of (1689, 409, 658)
    dataset_PLIF = file['PLIF']['PLIFfield'][:]

    # get the x range of PLIF image with (658, 1)
    PLIF_x = file['PLIF']['x'][:]

    # get the y range of PLIF image with (409, 1)
    PLIF_y = file['PLIF']['y'][:]


# PART 3: define the function for showing the mean image
def show_meanImage():
    mean_PIV = np.mean(dataset_PIV, axis=0)
    mean_PIV = np.transpose(mean_PIV)
    mean_PLIF = np.mean(dataset_PLIF, axis=0)
    mean_PLIF = np.mean(mean_PLIF)


# PART : the main part which calling the related functions for showing the datasets
show_meanImage()