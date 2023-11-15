import h5py
import numpy as np

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

# PART 3: crop the dataset
# 1. define the range of x, y
cropped_xmin = max(PLIF_x.min(), PIV_x.min())
cropped_ymin = max(PLIF_y.min(), PIV_y.min())
cropped_xmax = min(PLIF_x.max(), PIV_x.max())
cropped_ymax = min(PLIF_y.min(), PIV_y.min())

# 2. get the indices satisfied the range
indices_PLIF_x = np.where((PLIF_x >= cropped_xmin) & (PLIF_x <= cropped_xmax))[0]
indices_PLIF_y = np.where((PLIF_y >= cropped_ymin) & (PLIF_y <= cropped_ymax))[0]

indices_PIV_x = np.where((PIV_x >= cropped_xmin) & (PIV_x <= cropped_xmax))[0]
indices_PIV_y = np.where((PIV_y >= cropped_ymin) & (PIV_y <= cropped_ymax))[0]

# 3. crop the datasets via the range
cropped_PLIF = dataset_PLIF[indices_PLIF_y[:, np.newaxis], indices_PLIF_x]
cropped_PIV = dataset_PIV[:, indices_PIV_y[:, np.newaxis], indices_PIV_x]


