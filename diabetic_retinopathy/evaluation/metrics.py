import sklearn.metrics
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from itertools import cycle
from scipy import interp


class ConfusionMatrix(object):
    # Metrics of evaluation
    def __init__(self, original_prediction, y_pred, y_true, classification):

        '''
        Combine confusion matrix, roc curve and so on
        Args:
            original_prediction: the original probability of each label
            y_pred: list of predicted labels
            y_true: list of the ground truth
            classification: type of classification
        '''

        # initialize parameters
        # self.num_class = num_class
        self.y_pred = y_pred
        self.y_true = y_true
        self.classification = classification
        self.original_prediction = original_prediction

        cm = self.confusion_matrix_show(self.y_true, self.y_pred)
        self.confusion_matrix_plot(cm)
        self.roc_curve_plot(y_true, y_pred)
        if self.classification == 'binary':
            accuracy, precision, recall, f1_score, sensitivity, specificity = self.accuracy_for_binary_classification(cm)
            stats_text = "\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}\nSensitivity={:0.3f}\nSpecificity={:0.3f}"\
                .format(accuracy, precision, recall, f1_score, sensitivity, specificity)
        elif self.classification == 'multiple':
            accuracy, precision, recall, f1_score = self.accuracy_for_multiple_classifications(y_true, y_pred)
            stats_text = "\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}" \
                .format(accuracy, precision, recall, f1_score)
        print(stats_text)

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

    def accuracy_for_binary_classification(self, cm):

        '''
        Calculate the metrics using confusion matrix for binary classification
        Args:
            cm: confusion matrix
        Return:
            all the results of metrics
        '''

        accuracy = np.trace(cm) / float(np.sum(cm))
        precision = cm[1, 1] / sum(cm[1, :])
        recall = cm[1, 1] / sum(cm[1, :])
        f1_score = 2 * precision * recall / (precision + recall)
        sensitivity = cm[1, 1] / sum(cm[:, 1])
        specificity = cm[0, 0] / sum(cm[:, 0])
        return accuracy, precision, recall, f1_score, sensitivity, specificity

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
        if self.classification == 'binary':
            f1_score = sklearn.metrics.f1_score(y_true, y_pred)
        elif self.classification == 'multiple':
            f1_score = sklearn.metrics.f1_score(y_true, y_pred, average='macro')
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy, precision, recall, f1_score

    def confusion_matrix_plot(self, cm):

        '''
        Plot the image of confusion matrix
        Args:
            cm: confusion matrix
        '''

        plt.figure(dpi=120)
        group_counts = ["\n{0:0.0f}\n".format(value) for value in cm.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]
        if self.classification == 'binary':
            group_names = ['True_negative', 'False_negative', 'False_negative', 'True_positive']
            labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
            labels = np.asarray(labels).reshape(2, 2)
        elif self.classification == 'multiple':
            labels = [f"{v1}{v2}".strip() for v1, v2 in zip(group_counts, group_percentages)]
            labels = np.asarray(labels).reshape(5, 5)
        sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
        plt.xlabel('Predict')
        plt.ylabel('True')
        plt.title('Confusion matrix')
        plt.show()

    def roc_curve_plot(self, y_true, y_pred):

        '''
        Plot the image of ROC curve
        Args:
            y_true: list of ground truth
            y_pred: list of predicted labels
        '''

        # auc = roc_auc_score(y_test,clf.decision_function(X_test))
        if self.classification == 'binary':
            auc = roc_auc_score(y_true, y_pred)
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
            plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic example')
            plt.legend(loc="lower right")
            plt.show()

        elif self.classification == 'multiple':
            # Define the number of class
            indices = y_true
            depth = 5
            # Convert ground truth into one hot coding
            y_true_one_hot_coding = tf.one_hot(indices, depth, on_value=1, off_value=0, axis=-1)
            original_pred = self.original_prediction
            n_classes = 5
            # Calculate ROC curve for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true_one_hot_coding[:, i], original_pred[:, i])
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
            plt.figure()

            plt.plot(fpr["macro"], tpr["macro"],
                     label='macro-average ROC curve (area = {0:0.2f})'
                           ''.format(roc_auc["macro"]),
                     color='navy', linestyle=':', linewidth=4)

            colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'mediumorchid', 'goldenrod'])
            for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                         label='ROC curve of class {0} (area = {1:0.2f})'
                               ''.format(i, roc_auc[i]))

            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Some extension of Receiver operating characteristic to multi-class')
            plt.legend(loc="lower right")
            plt.show()
        return
