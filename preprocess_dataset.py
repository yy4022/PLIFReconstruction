from typing import Tuple, List

import h5py
import numpy as np
import matplotlib as mpl
import scipy.io
import math
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

"""
This file defines the functions for pre-processing the datasets.
"""

# Define the class of Dataset
class MyDataset(Dataset):
    def __init__(self, img_data):
        self.img_data = img_data
        self.length = len(self.img_data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.img_data[index]


def min_max_scaler(data: np.ndarray, min_value: float, max_value: float) -> np.ndarray:

    normalized_data = (data - min_value) / (max_value - min_value)
    return normalized_data

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


def preprocess_data(file_PIV: str, file_PLIF: str) \
        -> Tuple[np.ndarray, np.ndarray, float, float, float, float]:

    """
    pre-process the dataset provided by the new datasets
    :param file_PIV: a string represents the filename of PIV image
    :param file_PLIF: a string represents the filename of PLIF image
    :return: a numpy array represents the value of preprocessed PIV image,
            a numpy array represents the value of preprocessed PLIF image,
            four float numbers denotes xmin, xmax, ymin, ymax of preprocessed data
    """

    # STEP 1: load the datasets
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

    # STEP 2: crop the dataset
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

    # 4. change the type of dataset from 'float64' to 'float32'
    cropped_PIV = cropped_PIV.astype('float32')
    cropped_PLIF = cropped_PLIF.astype('float32')

    return cropped_PIV, cropped_PLIF, cropped_xmin, cropped_xmax, cropped_ymin, cropped_ymax

# Internal Function
def crop_data(image_data: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray)\
        -> np.ndarray:

    # STEP 1. define the range of x, y
    cropped_xmin = -15
    cropped_ymin = 0
    cropped_xmax = 15
    cropped_ymax = 30

    # STEP 2. get the indices satisfied the range
    indices_x = np.where((x_axis >= cropped_xmin) & (x_axis <= cropped_xmax))[0]
    indices_y = np.where((y_axis >= cropped_ymin) & (y_axis <= cropped_ymax))[0]

    # STEP 3. crop the dataset via the range
    cropped_data = image_data[:, indices_y[:, np.newaxis], indices_x]

    # # STEP3: change the type of dataset from 'float64' to 'float32'
    # cropped_PIV = cropped_PIV.astype('float32')
    # cropped_PLIF = cropped_PLIF.astype('float32')

    return cropped_data


def crop_old_PIVdata(files_PIV: List[str])\
        -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:

    cropped_PIV_x_data = []
    cropped_PIV_y_data = []
    cropped_PIV_z_data = []

    # loop to obtain the cropped PIV datasets
    for file_PIV in files_PIV:
        # STEP 1: load the PIV dataset
        PIV_information = scipy.io.loadmat(file_PIV)

        # 1.1. read the PIV dataset from the file
        dataset_PIV = PIV_information['PIV']['velfield'][:]
        dataset_PIV = np.concatenate([np.concatenate(sublist) for sublist in dataset_PIV])
        dataset_PIV = np.transpose(dataset_PIV)

        # 1.2. read the PIV x-axis
        PIV_x_axis = PIV_information['PIV']['x'][:]
        PIV_x_axis = np.concatenate([np.concatenate(sublist) for sublist in PIV_x_axis])
        PIV_x_axis = np.transpose(PIV_x_axis)

        # 1.3. read the PIV y-axis
        PIV_y_axis = PIV_information['PIV']['y'][:]
        PIV_y_axis = np.transpose(np.concatenate([np.concatenate(sublist) for sublist in PIV_y_axis]))
        # PIV_y = np.transpose(PIV_y)

        # STEP 2: crop the PIV dataset
        cropped_PIV_x = crop_data(image_data=dataset_PIV[0, :, :, :], x_axis=PIV_x_axis, y_axis=PIV_y_axis)
        cropped_PIV_y = crop_data(image_data=dataset_PIV[1, :, :, :], x_axis=PIV_x_axis, y_axis=PIV_y_axis)
        cropped_PIV_z = crop_data(image_data=dataset_PIV[2, :, :, :], x_axis=PIV_x_axis, y_axis=PIV_y_axis)

        # STEP 3: append the dataset to the corresponding list
        cropped_PIV_x_data.append(cropped_PIV_x)
        cropped_PIV_y_data.append(cropped_PIV_y)
        cropped_PIV_z_data.append(cropped_PIV_z)

    return cropped_PIV_x_data, cropped_PIV_y_data, cropped_PIV_z_data

def crop_old_PLIFdata(files_PLIF: List[str]) -> List[np.ndarray]:

    cropped_PLIF_data = []

    # loop to obtain the cropped PLIF datasets
    for file_PLIF in files_PLIF:
        # STEP 1: load the PLIF dataset
        with h5py.File(file_PLIF, 'r') as file:
            # get the PLIF numpy array
            dataset_PLIF = file['PLIF']['OH'][:]

            # get the x range of PLIF image with (504, 1)
            PLIF_x_axis = file['PLIF']['x'][:]

            # get the y range of PLIF image with (833, 1)
            PLIF_y_axis = file['PLIF']['y'][:]

        # STEP 2: crop the PLIF dataset
        cropped_PLIF = crop_data(image_data=dataset_PLIF, x_axis=PLIF_x_axis, y_axis=PLIF_y_axis)

        # STEP 3: append the dataset to the list
        cropped_PLIF_data.append(cropped_PLIF)

    return cropped_PLIF_data


def crop_old_data(file_PIV: str, file_PLIF: str) \
        -> Tuple[np.ndarray, np.ndarray]:

    """
       pre-process the dataset provided by the old datasets (used by Barwey)
       :param file_PIV: a string represents the filename of PIV image
       :param file_PLIF: a string represents the filename of PLIF image
       :return: a numpy array represents the value of preprocessed PIV image,
               a numpy array represents the value of preprocessed PLIF image
       """

    # STEP 1: load the PIV and PLIF dataset
    with h5py.File(file_PLIF, 'r') as file:
        # get the PLIF numpy array
        dataset_PLIF = file['PLIF']['OH'][:]

        # get the x range of PLIF image with (504, 1)
        PLIF_x = file['PLIF']['x'][:]

        # get the y range of PLIF image with (833, 1)
        PLIF_y = file['PLIF']['y'][:]

    PIV_information = scipy.io.loadmat(file_PIV)

    dataset_PIV = PIV_information['PIV']['velfield'][:]
    dataset_PIV = np.concatenate([np.concatenate(sublist) for sublist in dataset_PIV])
    dataset_PIV = np.transpose(dataset_PIV)

    PIV_x = PIV_information['PIV']['x'][:]
    PIV_x = np.concatenate([np.concatenate(sublist) for sublist in PIV_x])
    PIV_x = np.transpose(PIV_x)

    PIV_y = PIV_information['PIV']['y'][:]
    PIV_y = np.transpose(np.concatenate([np.concatenate(sublist) for sublist in PIV_y]))
    # PIV_y = np.transpose(PIV_y)

    # STEP 2: crop the dataset
    # 1. define the range of x, y
    cropped_xmin = -15
    cropped_ymin = 0
    cropped_xmax = 15
    cropped_ymax = 30

    # 2. get the indices satisfied the range
    indices_PLIF_x = np.where((PLIF_x >= cropped_xmin) & (PLIF_x <= cropped_xmax))[0]
    indices_PLIF_y = np.where((PLIF_y >= cropped_ymin) & (PLIF_y <= cropped_ymax))[0]

    indices_PIV_x = np.where((PIV_x >= cropped_xmin) & (PIV_x <= cropped_xmax))[0]
    indices_PIV_y = np.where((PIV_y >= cropped_ymin) & (PIV_y <= cropped_ymax))[0]

    # 3. crop the datasets via the range
    cropped_PIV = dataset_PIV[:, :, indices_PIV_y[:, np.newaxis], indices_PIV_x]
    cropped_PLIF = dataset_PLIF[:, indices_PLIF_y[:, np.newaxis], indices_PLIF_x]

    # # STEP3: change the type of dataset from 'float64' to 'float32'
    # cropped_PIV = cropped_PIV.astype('float32')
    # cropped_PLIF = cropped_PLIF.astype('float32')

    return cropped_PIV, cropped_PLIF

def preprocess_old_data(cropped_data: np.ndarray, min_value: float, max_value: float) \
        -> Tuple[np.ndarray, np.ndarray]:

    # STEP 1: normalize the image data via min-max scaling method
    normalized_data = min_max_scaler(cropped_data, min_value, max_value)

    # STEP 2: discretize the image data into 12 boxes (3 rows, 4 columns)
    discretized_data = discretize_image(normalized_data, rows=3, columns=4)

    # STEP 3: change the type of dataset from 'float64' to 'float32'
    discretized_data = discretized_data.astype('float32')

    return discretized_data

def discretize_image(image: np.ndarray, rows: int, columns: int):

    # STEP 1: get the size of x (last dimension) and y (second last dimension)
    x_size = image.shape[-1]
    y_size = image.shape[-2]
    image_num = image.shape[-3]
    img_num = 99

    # STEP 2: get the x_range and y_range of every box (total: 12 boxes)
    x_range = math.floor(x_size / columns)
    y_range = math.floor(y_size / rows)

    discretized_image = np.zeros((rows * columns, image_num, y_range, x_range))

    for j in range(rows):
        for i in range(columns):
            start_x = i * x_range
            end_x = (i + 1) * x_range
            start_y = j * y_range
            end_y = (j + 1) * y_range

            box = image[:, start_y:end_y, start_x:end_x]
            discretized_image[j * columns + i, :, :, :] = box

    return discretized_image

def show_box_PIV(image: np.ndarray, dimension: int, rows: int, columns: int):

    vmin = 0.0
    vmax = 1.0
    plt.figure(figsize=(16, 12))

    # i, j denotes the row and column of this box, respectively
    for j in range(rows):
        for i in range(columns):
            plt.subplot(3, 4, (rows - 1 - j) * columns + i + 1)

            if dimension == 1:
                plt.title(f'The box {(rows - 1 - j) * columns + i + 1} of PIV-x image')
            elif dimension == 2:
                plt.title(f'The box {(rows - 1 - j) * columns + i + 1} of PIV-y image')
            elif dimension == 3:
                plt.title(f'The box {(rows - 1 - j) * columns + i + 1} of PIV-z image')
            else:
                print("Error: the input dimension of PIV image is wrong.")
                exit()

            plt.imshow(image[j * columns + i][dimension - 1][:][:], cmap='turbo',
                       origin='lower', interpolation='bicubic', vmin=vmin, vmax=vmax)
            plt.colorbar(ticks=np.arange(vmin, vmax+0.1, 0.1))

    plt.show()


def show_box_PLIF(image: np.ndarray, rows: int, columns: int):

    vmin = 0.0
    vmax = 1.0
    plt.figure(figsize=(16, 12))

    # i, j denotes the row and column of this box, respectively
    for j in range(rows):
        for i in range(columns):
            plt.subplot(3, 4, (rows - 1 - j) * columns + i + 1)
            plt.title(f'The box {(rows - 1 - j) * columns + i + 1} of PLIF image')
            plt.imshow(image[j * columns + i][:][:], cmap='hot', origin='lower',
                       interpolation='bicubic', vmin=vmin, vmax=vmax)
            plt.colorbar(ticks=np.arange(vmin, vmax+0.1, 0.1))

    plt.show()

def show_PIV(image: np.ndarray, dimension: int):

    vmin = 0.0
    vmax = 1.0
    plt.figure(figsize=(16, 12))

    if dimension == 1:
        plt.title('The PIV-x image')
    elif dimension == 2:
        plt.title('The PIV-y image')
    elif dimension == 3:
        plt.title('The PIV-z image')
    else:
        print("Error: the input dimension of PIV image is wrong.")
        exit()

    plt.imshow(image[dimension-1][:][:], cmap='turbo', origin='lower', interpolation='bicubic',
               vmin=vmin, vmax=vmax)
    plt.colorbar(ticks=np.arange(vmin, vmax+0.1, 0.1))

    plt.show()

def show_PLIF(image: np.ndarray):

    vmin = 0.0
    vmax = 1.0
    plt.figure(figsize=(16, 12))

    plt.title('The PLIF image')
    plt.imshow(image, cmap='hot', origin='lower', interpolation='bicubic',
               vmin=vmin, vmax=vmax)

    plt.colorbar(ticks=np.arange(vmin, vmax+0.1, 0.1))

    plt.show()
