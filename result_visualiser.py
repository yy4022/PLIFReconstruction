from typing import Dict, List

import numpy as np
from matplotlib import pyplot as plt


# Internal Function
def show_boxes_image(image_data: np.ndarray, rows: int, columns: int, vmin: float, vmax: float,
                     filename: str, title: str):

    """
    Internal Function:
        Show the discretized image as the order of the box.
    :param image_data: A numpy array contains a given image data with several boxes.
    :param rows: An integer denotes the number of rows of this discretized image.
    :param columns: An integer denotes the number of columns of this discretized image.
    :param vmin: A float denotes the minimum value of the colorbar.
    :param vmax: A float denotes the maximum value of the colorbar.
    :param filename: A string contains the name for saving the image.
    :param title: A string denotes the title of this image.
    :return: None.
    """

    plt.figure(figsize=(16, 12))
    plt.suptitle(title, fontsize=24, y=0.95)

    for j in range(rows):
        for i in range(columns):
            plt.subplot(3, 4, (rows - 1 - j) * columns + i + 1)

            plt.title(f"box {j * columns + i + 1}")
            plt.imshow(image_data[j * columns + i][:][:], cmap='turbo', origin='lower',
                       interpolation='bicubic', vmin=vmin, vmax=vmax)
            plt.colorbar(ticks=np.arange(vmin, vmax + 0.1, 0.1))

    plt.savefig(f"./result/{filename}")
    plt.show()


def show_loss(loss: Dict[str, List[float]], filename: str):

    """
    Visualizes the training and validation loss over epochs.
    :param loss: A dictionary contains training and validation loss records.
    :param filename: A string contains the name for saving the plot.
    :return: None.
    """

    plt.figure(figsize=(10, 8))
    plt.semilogy(loss['train_loss_records'], label='Train')
    plt.semilogy(loss['validation_loss_records'], label='Valid')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.legend()

    plt.savefig(f"./result/{filename}")
    plt.show()


def show_difference(difference: np.ndarray, filename: str, rows: int, columns: int, img_num: int):

    """
    Visualizes the difference between the prediction and actual output image data of the specified image.
    :param difference: A numpy array contains the difference data.
    :param filename: A string contains the name for saving the image.
    :param rows: An integer denotes the number of rows of this discretized image.
    :param columns: An integer denotes the number of columns of this discretized image.
    :param img_num: An integer denotes the specified image number to show.
    :return: None.
    """

    vmin = 0.0
    vmax = 1.0
    title = "The difference of PIV-x " + str(img_num)

    show_boxes_image(image_data=difference[:, img_num, :, :], rows=rows, columns=columns,
                     vmin=vmin, vmax=vmax, filename=filename, title=title)


def show_comparison(prediction_data: np.ndarray, actual_data: np.ndarray, prediction_filename: str,
                    actual_filename: str, rows: int, columns: int, img_num: int):

    """
    Visualizes the prediction and actual output image data of the specified image for comparing.
    :param prediction_data: A numpy array contains the prediction data.
    :param actual_data: A numpy array contains the actual output.
    :param prediction_filename: A string contains the name for saving the prediction image.
    :param actual_filename: A string contains the name for saving the actual image.
    :param rows: An integer denotes the number of rows of this discretized image.
    :param columns: An integer denotes the number of columns of this discretized image.
    :param img_num: An integer denotes the specified image number to show.
    :return: None.
    """

    vmin = 0.0
    vmax = 1.0
    prediction_title = "The prediction data of PIV-x " + str(img_num)
    actual_title = "The actual data of PIV-x " + str(img_num)

    show_boxes_image(image_data=prediction_data[:, img_num, :, :], rows=rows, columns=columns,
                     vmin=vmin, vmax=vmax, filename=prediction_filename, title=prediction_title)

    show_boxes_image(image_data=actual_data[:, img_num, :, :], rows=rows, columns=columns,
                     vmin=vmin, vmax=vmax, filename=actual_filename, title=actual_title)


