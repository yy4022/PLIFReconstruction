import h5py
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

# PART 1: define the parameters
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

# print(f'min PLIF-x = {PLIF_x.min()}')
# print(f'max PLIF-x = {PLIF_x.max()}')
# print(f'min PIV-x = {PIV_x.min()}')
# print(f'max PIV-x = {PIV_x.max()}')
#
# print(f'min PLIF-y = {PLIF_y.min()}')
# print(f'max PLIF-y = {PLIF_y.max()}')
# print(f'min PIV-y = {PIV_y.min()}')
# print(f'max PIV-y = {PIV_y.max()}')

# print(np.shape(dataset_PIV))
# print(np.shape(dataset_PLIF))

# PART 3: crop the dataset
# 1. define the range of x, y
cropped_xmin = max(PLIF_x.min(), PIV_x.min())
cropped_ymin = max(PLIF_y.min(), PIV_y.min())
cropped_xmax = min(PLIF_x.max(), PIV_x.max())
cropped_ymax = min(PLIF_y.max(), PIV_y.max())

# 2. get the indices satisfied the range
indices_PLIF_x = np.where((PLIF_x >= cropped_xmin) & (PLIF_x <= cropped_xmax))[0]
indices_PLIF_y = np.where((PLIF_y >= cropped_ymin) & (PLIF_y <= cropped_ymax))[0]

indices_PIV_x = np.where((PIV_x >= cropped_xmin) & (PIV_x <= cropped_xmax))[0]
indices_PIV_y = np.where((PIV_y >= cropped_ymin) & (PIV_y <= cropped_ymax))[0]

# 3. crop the datasets via the range
cropped_PIV = dataset_PIV[:, :, indices_PIV_y[:, np.newaxis], indices_PIV_x]
cropped_PLIF = dataset_PLIF[:, indices_PLIF_y[:, np.newaxis], indices_PLIF_x]


# print(np.shape(cropped_PIV))
# print(np.shape(cropped_PLIF))
# print(np.shape(cropped_PIV[:, 999, :, :]))
# print(np.shape(cropped_PLIF[999, :, :]))

# APPENDIX: related functions used in this file
def show_image(PIV_image: np.ndarray, PLIF_image: np.ndarray,
               xmin: float, xmax: float, ymin: float, ymax: float) -> None:
    """
    show the given PIV and PLIF image
    :param ymax: a float number represents the maximum value of y-axis
    :param ymin: a float number represents the minimum value of y-axis
    :param xmax: a float number represents the maximum value of x-axis
    :param xmin: a float number represents the minimum value of x-axis
    :param PIV_image: a numpy array with 3 dimensions representing the value of each pixel in PIV-x, y, z images
    :param PLIF_image: a numpy array with 2 dimensions representing the value of each pixel in PLIF image
    """
    plt.figure(figsize=(16, 8))
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

    # show the given PLIF image
    plt.subplot(1, 4, 1)
    plt.title('The PLIF Image')
    plt.imshow(PLIF_image, cmap='hot', extent=(xmin, xmax, ymin, ymax),
               origin='lower', interpolation='bicubic')
    sm = plt.cm.ScalarMappable(norm=norm)
    sm.set_array([])
    plt.colorbar()

    # show the given PIV-x image
    plt.subplot(1, 4, 2)
    plt.title('The PIV-x Image')
    plt.imshow(PIV_image[0][:][:], cmap='turbo', extent=(xmin, xmax, ymin, ymax),
               origin='lower', interpolation='bicubic')
    sm = plt.cm.ScalarMappable(norm=norm)
    sm.set_array([])
    plt.colorbar()

    # show the mean PIV-y image
    plt.subplot(1, 4, 3)
    plt.title('The PIV-y Image')
    plt.imshow(PIV_image[1][:][:], cmap='turbo', extent=(xmin, xmax, ymin, ymax),
               origin='lower', interpolation='bicubic')
    sm = plt.cm.ScalarMappable(norm=norm)
    sm.set_array([])
    plt.colorbar()

    # show the mean PIV-z image
    plt.subplot(1, 4, 4)
    plt.title('The PIV-z Image')
    plt.imshow(PIV_image[2][:][:], cmap='turbo', extent=(xmin, xmax, ymin, ymax),
               origin='lower', interpolation='bicubic')
    sm = plt.cm.ScalarMappable(norm=norm)
    sm.set_array([])
    plt.colorbar()

    plt.show()


show_image(cropped_PIV[:, 999, :, :], cropped_PLIF[999, :, :],
           cropped_xmin, cropped_xmax, cropped_ymin, cropped_ymax)
