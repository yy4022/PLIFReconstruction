import h5py
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

"""
This file is provided for visualising the datasets for training, validating and testing model
"""

# PART 1: define the parameters
first_PLIF_num = int(1)
show_num = 1000

file_PIV = str('data/IA_PIV.mat')
file_PLIF = str('data/IA_PLIF_1to2500.mat')

# PART 2: load the datasets
with h5py.File(file_PIV, 'r') as file:

    """
    Get the PIV numpy array with the shape of (3, 6689, 73, 73), note that:
        1 denotes the axial(x) velocity, 
        2 denotes the radial(y) velocity, 
        3 denotes the tangential(z) velocity
    """
    dataset_PIV = file['PIV']['velfield'][:]

    # get the x range of PIV image with (73, 1)
    PIV_x = file['PIV']['x'][:]

    # get the y range of PIV image with (73, 1)
    PIV_y = file['PIV']['y'][:]

with h5py.File(file_PLIF, 'r') as file:

    # get the PLIF numpy array with the shape of (1689, 409, 658)
    dataset_PLIF = file['PLIF']['PLIFfield'][:]

    # get the x range of PLIF image with (658, 1)
    PLIF_x = file['PLIF']['x'][:]

    # get the y range of PLIF image with (409, 1)
    PLIF_y = file['PLIF']['y'][:]


# PART 3: define the function for showing the mean image
def show_meanImage():
    mean_PIV = np.mean(dataset_PIV, axis=1)  # (3, 73, 73)
    mean_PLIF = np.mean(dataset_PLIF, axis=0) # (608, 409)

    plt.figure(figsize=(16, 8))
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

    # show the mean PLIF image
    plt.subplot(1, 4, 1)
    plt.title('The Mean Image of PLIF')
    plt.imshow(mean_PLIF, cmap='hot', extent=(PLIF_x.min(), PLIF_x.max(), PLIF_y.min(), PLIF_y.max()),
               origin='lower', interpolation='bicubic')
    sm = plt.cm.ScalarMappable(norm=norm)
    sm.set_array([])
    plt.colorbar()

    # show the mean PIV-x image
    plt.subplot(1, 4, 2)
    plt.title('The Mean Image of PIV-x')
    plt.imshow(mean_PIV[0][:][:], cmap='turbo', extent=(PIV_x.min(), PIV_x.max(), PIV_y.min(), PIV_y.max()),
               origin='lower', interpolation='bicubic')
    sm = plt.cm.ScalarMappable(norm=norm)
    sm.set_array([])
    plt.colorbar()

    # show the mean PIV-y image
    plt.subplot(1, 4, 3)
    plt.title('The Mean Image of PIV-y')
    plt.imshow(mean_PIV[1][:][:], cmap='turbo', extent=(PIV_x.min(), PIV_x.max(), PIV_y.min(), PIV_y.max()),
               origin='lower', interpolation='bicubic')
    sm = plt.cm.ScalarMappable(norm=norm)
    sm.set_array([])
    plt.colorbar()

    # show the mean PIV-z image
    plt.subplot(1, 4, 4)
    plt.title('The Mean Image of PIV-z')
    plt.imshow(mean_PIV[2][:][:], cmap='turbo', extent=(PIV_x.min(), PIV_x.max(), PIV_y.min(), PIV_y.max()),
               origin='lower', interpolation='bicubic')
    sm = plt.cm.ScalarMappable(norm=norm)
    sm.set_array([])
    plt.colorbar()

    plt.show()

# PART 4: define the function for showing the specified image
def show_specifiedImage():
    plt.figure(figsize=(16, 8))
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

    # show the specified PLIF image
    plt.subplot(1, 4, 1)
    plt.title(f'The PLIF Image {show_num}')
    plt.imshow(dataset_PLIF[show_num-first_PLIF_num][:][:], cmap='hot',
               extent=(PLIF_x.min(), PLIF_x.max(), PLIF_y.min(), PLIF_y.max()), origin='lower',
               interpolation='bicubic')
    sm = plt.cm.ScalarMappable(norm=norm)
    sm.set_array([])
    plt.colorbar()

    # show the mean PIV-x image
    plt.subplot(1, 4, 2)
    plt.title('The Mean Image of PIV-x')
    plt.imshow(dataset_PIV[0][show_num-1][:][:], cmap='turbo',
               extent=(PIV_x.min(), PIV_x.max(), PIV_y.min(), PIV_y.max()), origin='lower',
               interpolation='bicubic')
    sm = plt.cm.ScalarMappable(norm=norm)
    sm.set_array([])
    plt.colorbar()

    # show the mean PIV-y image
    plt.subplot(1, 4, 3)
    plt.title('The Mean Image of PIV-y')
    plt.imshow(dataset_PIV[1][show_num-1][:][:], cmap='turbo',
               extent=(PIV_x.min(), PIV_x.max(), PIV_y.min(), PIV_y.max()), origin='lower',
               interpolation='bicubic')
    sm = plt.cm.ScalarMappable(norm=norm)
    sm.set_array([])
    plt.colorbar()

    # show the mean PIV-z image
    plt.subplot(1, 4, 4)
    plt.title('The Mean Image of PIV-z')
    plt.imshow(dataset_PIV[2][show_num-1][:][:], cmap='turbo',
               extent=(PIV_x.min(), PIV_x.max(), PIV_y.min(), PIV_y.max()), origin='lower',
               interpolation='bicubic')
    sm = plt.cm.ScalarMappable(norm=norm)
    sm.set_array([])
    plt.colorbar()

    plt.show()


# PART : the main part which calling the related functions for showing the datasets
show_meanImage()
show_specifiedImage()