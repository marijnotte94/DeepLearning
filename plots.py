import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from scipy import interp
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

def plot_confusion_matrix(figures_dir, filename, test_gen, test_length, label_encoder, predictions, show_values=False):
    if not os.path.isdir(figures_dir):
        os.makedirs(figures_dir)

    # get true labels from generator
    y_true = []
    for i in range(test_length):
        _, label = next(test_gen)
        y_true.append(label)

    # inverse transform true labels
    y_true = np.array(y_true).argmax(axis=-1)
    y_true = label_encoder.inverse_transform(y_true)

    # inverse transform prediction labels
    y_pred = predictions.argmax(axis=-1)
    y_pred = label_encoder.inverse_transform(y_pred)

    # create normalized confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(label_encoder.classes_))
    plt.xticks(tick_marks, list(label_encoder.classes_), rotation=90)
    plt.yticks(tick_marks, list(label_encoder.classes_))

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

def plot_ROC(figures_dir, filename, test_gen, test_length, label_encoder, predictions):
    if not os.path.isdir(figures_dir):
        os.makedirs(figures_dir)

    num_classes = len(label_encoder.classes_)

    # get true labels from generator
    y_true = []
    for i in range(test_length):
        _, label = next(test_gen)
        y_true.append(label)

    y_true = np.array(y_true).reshape((-1, num_classes))

    # get predictions
    y_pred = predictions

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= num_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle='-', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle='-', linewidth=4)

    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    # for i, color in zip(range(num_classes), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
    #              label='ROC curve of class {0} (area = {1:0.2f})'
    #                    ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(figures_dir, filename), ext='png', dpi=150)
    plt.show()

    return roc_auc["micro"], roc_auc["macro"]