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
