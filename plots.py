import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix
import os

def plot_accuracy(figures_dir, filename, history):
    if not os.path.isdir(figures_dir):
        os.makedirs(figures_dir)

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(figures_dir, filename), ext='png', dpi=150)
    plt.show()

def plot_loss(figures_dir, filename, history):
    if not os.path.isdir(figures_dir):
        os.makedirs(figures_dir)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(figures_dir, filename), ext='png', dpi=150)
    plt.show()

def plot_confusion_matrix(figures_dir, filename, generator, predictions, show_values=False):
    if not os.path.isdir(figures_dir):
        os.makedirs(figures_dir)

    classes = generator.class_indices.keys()

    y_true = generator.classes
    y_pred = predictions.argmax(axis=-1)

    # create normalized confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, list(classes), rotation=90)
    plt.yticks(tick_marks, list(classes))

    if show_values:
        fmt = '.2f'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(figures_dir, filename), ext='png', dpi=150)
    plt.show()