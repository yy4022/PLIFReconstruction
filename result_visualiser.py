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

def show_difference(difference: np.ndarray, filename: str, rows: int, columns: int, img_num: int):

    vmin = 0.0
    vmax = 1.0
    plt.figure(figsize=(16, 12))

    for j in range(rows):
        for i in range(columns):
            plt.subplot(3, 4, (rows - 1 - j) * columns + i + 1)

            plt.title(f"The difference of PIV-x {img_num}")
            plt.imshow(difference[j * columns + i][img_num][:][:], cmap='turbo', origin='lower', interpolation='bicubic',
                       vmin=vmin, vmax=vmax)
            plt.colorbar(ticks=np.arange(vmin, vmax+0.1, 0.1))

    plt.savefig(f"./result/{filename}")
    plt.show()

