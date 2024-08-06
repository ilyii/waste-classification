import numpy as np
import matplotlib.pyplot as plt
import torch


def display(img):
    MEAN = torch.tensor([0.485, 0.456, 0.406])
    STD = torch.tensor([0.229, 0.224, 0.225])

    img = img * STD[:, None, None] + MEAN[:, None, None]
    #img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    return npimg


def plot_validation_loss(val_loss, p):
    epochs = range(1, len(val_loss) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, val_loss, label='Validation Loss', marker='o')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(p)

def plot_validation_accuracy(val_acc, p):
    epochs = range(1, len(val_acc) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, val_acc, label='Validation Accuracy', marker='o')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(p)

def plot_training_loss(train_loss, p):
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_loss, label='Training Loss', marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(p)

def plot_training_accuracy(train_acc, p):
    epochs = range(1, len(train_acc) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_acc, label='Training Accuracy', marker='o')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(p)

def plot_time_per_epoch(time_per_epoch, p):
    epochs = range(1, len(time_per_epoch) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, time_per_epoch, label='Time per Epoch', marker='o', color='purple')
    plt.title('Time per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.savefig(p)


def plot_confusion_matrix(y_true, y_pred, classes):
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('output.png')