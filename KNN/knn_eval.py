from functions.functions import *
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import confusion_matrix

# define k for the k-nearest-neighbours
k = 3

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

    # split data into training and test data
    samples_train, samples_test, labels_train, labels_test = train_test_split(samples,
                                                                              labels,
                                                                              test_size=0.2,
                                                                              random_state=42,
                                                                              shuffle=True)

    # predict labels with self-built KNN-method
    labels_pred_own = knn_predict(k=k,
                                  samples_train=samples_train,
                                  samples_test=samples_test,
                                  labels_train=labels_train)

    # predict labels with built-in KNN-method
    knn_builtin = KNeighborsClassifier(n_neighbors=k)
    knn_builtin.fit(samples_train, labels_train)
    labels_pred_builtin = knn_builtin.predict(samples_test)

    # calculate true pos, true-neg, false-pos and false-neg for metric calculation
    # for self-built and build-in KNN
    tp, tn, fp, fn = get_TP_TN_FP_FN(labels_pred_own, labels_test)
    tp_builtin, tn_builtin, fp_builtin, fn_builtin = get_TP_TN_FP_FN(labels_pred_builtin, labels_test)

    # calculate metrics for self-built and built-in KNN
    accuracy, precision, recall, f1 = get_acc_precision_recall_f1(tp=tp, tn=tn, fp=fp, fn=fn)
    accuracy_builtin, precision_builtin, recall_builtin, f1_builtin = get_acc_precision_recall_f1(tp=tp_builtin, tn=tn_builtin, fp=fp_builtin, fn=fn_builtin)

    # store metrics in a dictionary -> easier to store
    metrics = {'Accuracy': (accuracy, accuracy_builtin),
               'Precision': (precision, precision_builtin),
               'Recall': (recall, recall_builtin),
               'F1-Score': (f1, f1_builtin)}

    #print metrics to console
    print_metrics(metrics=metrics)

    # calculate confusion matrices for visualization tasks
    confmat_own = confusion_matrix(labels_test, labels_pred_own)
    confmat_builtin = confusion_matrix(labels_test, labels_pred_builtin)
    confmat_title_own = f'Confusion Matrix for {dataframe} for own KNN'
    confmat_title_builtin = f'Confusion Matrix for {dataframe} for built-in KNN'

    # plot confusion matrices for comparison between self-built and built-in KNN
    plt.subplot(121)
    show_confusion_matrix(confusion_matrix=confmat_own, title=confmat_title_own)
    plt.subplot(122)
    show_confusion_matrix(confusion_matrix=confmat_builtin, title=confmat_title_builtin)
    plt.show()
