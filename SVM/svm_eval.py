from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from functions.functions import *

import numpy as np

from functions.functions import read_csv_as_np, get_labels_and_samples


class SVM:

    def __init__(self, C=1.0):
        # C = error term
        self.C = C
        self.w = 0
        self.b = 0

    # Hinge Loss Function / Calculation
    def hingeloss(self, w, b, x, y):
        # Regularizer term
        reg = 0.5 * (w * w)

        for i in range(x.shape[0]):
            # Optimization term
            opt_term = y[i] * ((np.dot(w, x[i])) + b)

            # calculating loss
            loss = reg + self.C * max(0, 1 - opt_term)
        return loss[0][0]

    def fit(self, X, Y, batch_size=200, learning_rate=0.01, epochs=500):
        n_samples, n_features = X.shape
        c = self.C
        idx = np.arange(n_samples)
        np.random.shuffle(idx)
        w = np.zeros((1, n_features))
        b = 0
        losses = []
        for i in range(epochs):
            l = self.hingeloss(w, b, X, Y)
            losses.append(l)
            for batch in range(0, n_samples, batch_size):
                gradient_w = 0
                gradient_b = 0
                for j in range(batch, batch + batch_size):
                    if j < n_samples:
                        x = idx[j]
                        ti = Y[x] * (np.dot(w, X[x].T) + b)
                        if ti > 1:
                            gradient_w += 0
                            gradient_b += 0
                        else:
                            gradient_w += c * Y[x] * X[x]
                            gradient_b += c * Y[x]
                w = w - learning_rate * w + learning_rate * gradient_w
                b = b + learning_rate * gradient_b
        self.w = w
        self.b = b

        return self.w, self.b, losses

    def predict(self, X):

        prediction = np.dot(X, self.w[0]) + self.b  # w.x + b
        return np.sign(prediction)

# Paths to dataframes
path_df_bottle_damaged = '../dataframes/_bottle_damaged/dataframe.csv'
path_df_label_damaged = '../dataframes/_label_damaged/dataframe.csv'
path_df_open_closed = '../dataframes/_open_closed/dataframe.csv'

# Load dataframes into np.arrays because of better performance compared to pd.Dataframes
df_bottle_damaged = read_csv_as_np(path_df_bottle_damaged)
df_label_damaged = read_csv_as_np(path_df_label_damaged)
df_open_closed = read_csv_as_np(path_df_open_closed)

# Store dataframes in a dictionary because of the following iteration task
dataframes = {'bottle_damaged': df_bottle_damaged,
              'label_damaged': df_label_damaged,
              'open_closed': df_open_closed}

for dataframe in dataframes:
    print(dataframe)
    # split the data into labels and samples
    labels, samples = get_labels_and_samples(dataframes[dataframe])
    labels = np.where(labels == 1, 1, -1)

    # split data into training and test data
    samples_train, samples_test, labels_train, labels_test = train_test_split(samples,
                                                                              labels,
                                                                              test_size=0.2,
                                                                              random_state=42,
                                                                              shuffle=True)

    svm_own = SVM()
    svm_builtin = SVC()

    w, b, losses = svm_own.fit(samples_train, labels_train)
    svm_builtin.fit(samples_train, labels_train)

    labels_pred_own = svm_own.predict(samples_test)
    labels_pred_builtin = svm_builtin.predict(samples_test)

    labels_pred_builtin = np.where(labels_pred_builtin == 1, 1, 2)
    labels_pred_own = np.where(labels_pred_own == 1, 1, 2)
    labels_test = np.where(labels_test == 1, 1, 2)

    tp, tn, fp, fn = get_TP_TN_FP_FN(labels_pred_own, labels_test)
    tp_builtin, tn_builtin, fp_builtin, fn_builtin = get_TP_TN_FP_FN(labels_pred_builtin, labels_test)

    # calculate metrics for self-built and built-in GNB
    accuracy, precision, recall, f1 = get_acc_precision_recall_f1(tp=tp, tn=tn, fp=fp, fn=fn)
    accuracy_builtin, precision_builtin, recall_builtin, f1_builtin = get_acc_precision_recall_f1(tp=tp_builtin,
                                                                                                  tn=tn_builtin,
                                                                                                  fp=fp_builtin,
                                                                                                  fn=fn_builtin)

    # store metrics in a dictionary -> easier to store
    metrics = {'Accuracy': (accuracy, accuracy_builtin),
               'Precision': (precision, precision_builtin),
               'Recall': (recall, recall_builtin),
               'F1-Score': (f1, f1_builtin)}
    # print metrics to console
    print_metrics(metrics=metrics)

    # calculate confusion matrices for visualization tasks
    confmat_own = confusion_matrix(labels_test, labels_pred_own)
    confmat_builtin = confusion_matrix(labels_test, labels_pred_builtin)
    confmat_title_own = f'Confusion Matrix for {dataframe} for own SVM'
    confmat_title_builtin = f'Confusion Matrix for {dataframe} for built-in SVM'

    # plot confusion matrices for comparison between self-built and built-in SVM
    plt.subplot(121)
    show_confusion_matrix(confusion_matrix=confmat_own, title=confmat_title_own)
    plt.subplot(122)
    show_confusion_matrix(confusion_matrix=confmat_builtin, title=confmat_title_builtin)
    plt.show()
