import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def plot_confusion_matrix(cm, class_names, title='Confusion Matrix'):
    if 'Entity' in title:
        figure = plt.figure(figsize=(40, 40))
    elif 'Intent' in title:
        figure = plt.figure(figsize=(35, 35))
    else:
        figure = plt.figure(figsize=(13, 13))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix
    cm = np.around(cm.astype('float') / cm.sum(axis=1)
                   [:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def get_confusion_matrix(y_hat, targets, enc):
    y_hat = enc.inverse_transform(y_hat)
    targets = enc.inverse_transform(targets)
    class_names = enc.classes_

    cm = confusion_matrix(y_hat, targets, labels=class_names)
    if len(class_names) == 57:
        title = 'Entity Confusion Matrix'
    elif len(class_names) == 54:
        title = 'Intent Confusion Matrix'
    else:
        title = 'Scenario Confusion Matrix'

    fig = plot_confusion_matrix(cm, class_names, title)
    return fig


def get_precision_recall(y_hat, targets):
    targets = targets
    percision, recall, fs, _ = precision_recall_fscore_support(
        y_hat, targets, average='micro')
    return percision, recall, fs


def classifcation_report(y_hat, targets, enc):
    percision, recall, fs = get_precision_recall(
        y_hat.flatten(), targets.flatten())
    fig = get_confusion_matrix(y_hat.flatten(), targets.flatten(), enc)
    return percision, recall, fs, fig
