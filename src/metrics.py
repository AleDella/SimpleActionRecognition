import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score

def plotConfusionMatrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    '''
    Function that plots the confusion matrix given predictions and truths

    Args:
        y_true (numpy.array): true labels
        y_pred (numpy.array): predicted labels
        classes (list[str]): names of the classes
        normalize (bool): flag to normalize the counts by rows
        title (str): title of the plot
        cmap: colormap for the plot
    Returns:
        ax: axis of the plot
    '''
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def findAccuracy(true, predictions):
    acc = accuracy_score(true, predictions)
    print ('accuracy score: %0.3f' % acc)
    return acc
