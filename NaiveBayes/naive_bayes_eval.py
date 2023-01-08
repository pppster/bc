from functions.functions import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.metrics import confusion_matrix

path_df_bottle_damaged = '../dataframes/_bottle_damaged/dataframe.csv'
path_df_label_damaged = '../dataframes/_label_damaged/dataframe.csv'
path_df_open_closed = '../dataframes/_open_closed/dataframe.csv'

df_bottle_damaged = read_csv_as_np(path_df_bottle_damaged)
df_label_damaged = read_csv_as_np(path_df_label_damaged)
df_open_closed = read_csv_as_np(path_df_open_closed)

dataframes = {'bottle_damaged': df_bottle_damaged,
              'label_damaged': df_label_damaged,
              'open_closed': df_open_closed}

for dataframe in dataframes:
    print(dataframe)
    labels, samples = get_labels_and_samples(dataframes[dataframe])
    samples_train, samples_test, labels_train, labels_test = train_test_split(samples,
                                                                              labels,
                                                                              test_size=0.2,
                                                                              random_state=42,
                                                                              shuffle=True)
    model = GaussianNB()
    model.fit(X=samples_train, y=labels_train)
    labels_pred = model.predict(samples_test)

    tp, tn, fp, fn = get_TP_TN_FP_FN(labels_pred, labels_test)
    accuracy, precision, recall, f1 = get_acc_precision_recall_f1(tp=tp, tn=tn, fp=fp, fn=fn)
    metrics = {'Accuracy': accuracy,
               'Precision': precision,
               'Recall': recall,
               'F1-Score': f1}
    print_metrics_cnn(metrics=metrics)
    confmat = confusion_matrix(labels_test, labels_pred)
    confmat_title = f'Confusion Matrix for {dataframe} with Naive Bayes'
    show_confusion_matrix(confusion_matrix=confmat, title=confmat_title)
    plt.show()
