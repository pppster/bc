from sklearn.model_selection import train_test_split
from functions.functions import *


from sklearn.naive_bayes import GaussianNB

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

    # setup built-in GNB
    model = GaussianNB()
    model.fit(X=samples_train, y=labels_train)
    labels_pred_builtin = model.predict(samples_test)

    # fit self-built GNB
    priories, distributions = fit_naivebayes(labels_train=labels_train, samples_train=samples_train)

    # predict with self-built GNB
    labels_pred_own = []
    for sample in samples_test:
        probabilities = get_probabilities(sample=sample, priories=priories, distributions=distributions)
        labels_pred_own.append(probabilities)

    # encode labels for comparison
    labels_pred_own = np.round(np.array(labels_pred_own))
    labels_pred_own = np.argmax(labels_pred_own, axis=1) + 1
    labels_test = get_one_hot_encoding(labels_test)
    labels_test = np.argmax(labels_test, axis=1) + 1

    # calculate true pos, true-neg, false-pos and false-neg for metric calculation
    # for self-built and build-in GNB
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
    confmat_title_own = f'Confusion Matrix for {dataframe} for own GNB'
    confmat_title_builtin = f'Confusion Matrix for {dataframe} for built-in GNB'

    # plot confusion matrices for comparison between self-built and built-in GNB
    plt.subplot(121)
    show_confusion_matrix(confusion_matrix=confmat_own, title=confmat_title_own)
    plt.subplot(122)
    show_confusion_matrix(confusion_matrix=confmat_builtin, title=confmat_title_builtin)
    plt.show()
