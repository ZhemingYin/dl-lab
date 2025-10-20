import logging
import os
import gin
import sklearn.metrics
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from itertools import cycle
from scipy import interp


@gin.configurable
class ConfusionMatrix(object):
    def __init__(self, original_prediction, y_pred, y_true, label_type, name):

        '''
        Combine confusion matrix, roc curve and so on
        Args:
            original_prediction: the original probability of each label
            y_pred: list of predicted labels
            y_true: list of the ground truth
            label_type: type of label, s2s or s2l
            name: name of dataset, HAPT or HAR
        '''

        # initialize parameters
        # self.num_class = num_class
        self.y_pred = y_pred
        self.y_true = y_true
        self.original_prediction = original_prediction
        self.label_type = label_type
        self.name = name

        cm = self.confusion_matrix_show(self.y_true, self.y_pred)
        self.confusion_matrix_plot(cm)
        accuracy, precision, recall, f1_score = self.accuracy_for_multiple_classifications(y_true, y_pred)
        stats_text = "\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}" \
            .format(accuracy, precision, recall, f1_score)
        print(stats_text)
        self.roc_curve_plot(y_true)


    def confusion_matrix_show(self, y_true, y_pred):

        '''
        Calculate the confusion matrix
        Args:
            y_true: list of ground truth
            y_pred: list of predicted labels
        Return:
            the confusion matrix
        '''

        cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
        print(cm)
        return cm

    def accuracy_for_multiple_classifications(self, y_true, y_pred):

        '''
        Calculate the metrics using confusion matrix for multiple classification
        Args:
            y_true: list of ground truth
            y_pred: list of predicted labels
        Return:
            all the metrics suitable for multiple classification
        '''

        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1_score = sklearn.metrics.f1_score(y_true, y_pred, average='macro')
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy, precision, recall, f1_score

    def confusion_matrix_plot(self, cm):

        '''
        Plot the image of confusion matrix
        Args:
            cm: confusion matrix
        '''

        plt.figure(figsize=[10, 10], dpi=200)
        group_counts = ["\n{0:0.0f}\n".format(value) for value in cm.flatten()]
        group_percentages = ["{0:.1%}".format(value) for value in np.ndarray.flatten(cm / np.sum(cm, axis=1))]
        labels = [f"{v1}{v2}".strip() for v1, v2 in zip(group_counts, group_percentages)]
        if self.label_type == 's2s':
            sns.heatmap(cm, annot=labels, fmt='', cmap='PuOr', annot_kws={"fontsize": 10}, vmax=5000, vmin=-1000, center=0, cbar=False)
        elif self.label_type == 's2l':
            sns.heatmap(cm, annot=labels, fmt='', cmap='PuOr', annot_kws={"fontsize": 10}, vmax=30, vmin=-10, center=0, cbar=False)
        else:
            raise ValueError
        plt.xlabel('Predict')
        plt.ylabel('True')
        if self.name == 'HAPT':
            plt.xticks(np.arange(0.5, 12.5, step=1),
                       labels=['W', 'WU', 'WD', 'SI', 'ST', 'L', 'ST2SI', 'SI2ST', 'SI2L', 'L2SI', 'ST2L', 'L2ST'],
                       rotation=45)
            plt.yticks(np.arange(0.5, 12.5, step=1),
                       labels=['W', 'WU', 'WD', 'SI', 'ST', 'L', 'ST2SI', 'SI2ST', 'SI2L', 'L2SI', 'ST2L', 'L2ST'],
                       rotation=45)
        else:
            pass
        plt.title('Confusion matrix')
        plt.savefig(os.getcwd() + '/visualization/cm.png')
        plt.show()

    def roc_curve_plot(self, y_true):

        '''
        Plot the image of ROC curve
        Args:
            y_true: list of ground truth
        '''

        # auc = roc_auc_score(y_test,clf.decision_function(X_test))
        # Define the number of classes
        indices = np.subtract(y_true, 1)
        # print(indices.shape)
        if self.name == 'HAPT':
            depth = 12
            n_classes = 12
        elif self.name == 'HAR':
            depth = 8
            n_classes = 8
        else:
            raise ValueError
        # Convert ground truth into one hot coding
        y_true_one_hot_coding = tf.one_hot(indices, depth)
        # print(y_true_one_hot_coding.shape)
        # Calculate ROC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_one_hot_coding[:, i], self.original_prediction[:, i])
            roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])

        # Compute macro-average ROC curve and ROC area
        # Aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = sklearn.metrics.auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        lw = 2
        plt.figure(dpi=200)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'mediumorchid', 'goldenrod', 'olive', 'teal',
                        'mediumorchid', 'navy', 'red', 'goldenrod', 'black'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i+1, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC to multi-class')
        plt.legend(loc="lower right")
        plt.savefig(os.getcwd() + '/visualization/roc.png')
        plt.show()
        return