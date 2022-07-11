import matplotlib.pyplot as plt
import seaborn
seaborn.set_style("darkgrid")


class MetricsUtils:
    def __init__(self):
        ...

    @staticmethod
    def plot_metrics(num_epochs,
                     train_loss_history,
                     val_loss_history,
                     train_acc_history,
                     val_acc_history):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

        ax[0].plot(range(1, num_epochs + 1), train_loss_history, label="Training")
        ax[0].plot(range(1, num_epochs + 1), val_loss_history, label="Validation")

        ax[1].plot(range(1, num_epochs + 1), train_acc_history, label="Training")
        ax[1].plot(range(1, num_epochs + 1), val_acc_history, label="Validation")

        ax[0].set_title("Training Metrics", fontsize=15)
        ax[0].set_xlabel("Epochs", fontsize=15)
        ax[0].set_ylabel("Loss", fontsize=15)
        ax[0].legend()
        ax[1].set_title("Validation Metrics", fontsize=15)
        ax[1].set_xlabel("Epochs", fontsize=15)
        ax[1].set_ylabel("Accuracy", fontsize=15)
        ax[1].legend()
