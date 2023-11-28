import numpy as np
from matplotlib import pyplot as plt


def show_loss(loss, filename: str):

    plt.figure(figsize=(10, 8))
    plt.semilogy(loss['train_loss_records'], label='Train')
    plt.semilogy(loss['validation_loss_records'], label='Valid')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.legend()

    plt.savefig(f"./result/{filename}")
    plt.show()

# Internal Function
def show_boxes_image(image_data: np.ndarray, rows: int, columns: int, vmin: float, vmax: float,
                     filename: str, title: str):

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


def show_difference(difference: np.ndarray, filename: str, rows: int, columns: int, img_num: int):

    vmin = 0.0
    vmax = 1.0
    title = "The difference of PIV-x " + str(img_num)

    show_boxes_image(image_data=difference[:, img_num, :, :], rows=rows, columns=columns,
                     vmin=vmin, vmax=vmax, filename=filename, title=title)


def show_comparison(prediction_data: np.ndarray, actual_data: np.ndarray, prediction_filename: str,
                    actual_filename: str, rows: int, columns: int, img_num: int):

    vmin = 0.0
    vmax = 1.0
    prediction_title = "The prediction data of PIV-x " + str(img_num)
    actual_title = "The actual data of PIV-x " + str(img_num)

    show_boxes_image(image_data=prediction_data[:, img_num, :, :], rows=rows, columns=columns,
                     vmin=vmin, vmax=vmax, filename=prediction_filename, title=prediction_title)

    show_boxes_image(image_data=actual_data[:, img_num, :, :], rows=rows, columns=columns,
                     vmin=vmin, vmax=vmax, filename=actual_filename, title=actual_title)






