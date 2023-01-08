import csv
import pandas as pd
import numpy as np
from numpy.linalg.linalg import norm
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from bci.core.bcimage import BCImage
from bci.utils.bcimageenum import BCImageEnum
from bci.utils.image_utils import just_image_names

def read_csv_as_np(path):
    with open(path, 'r') as f:
        data = list(csv.reader(f, delimiter=","))
    data = np.array(data)
    return data

def get_labels_and_samples(data: np.array):
    labels = np.array(data[1:, -1], dtype=int)
    samples = np.array(data[1:, 1:-1], dtype=float)
    return labels, samples

def get_TP_TN_FP_FN(labels_pred, labels_test):
    tp = np.sum(np.logical_and(labels_pred == 1, labels_test == 1))
    tn = np.sum(np.logical_and(labels_pred == 2, labels_test == 2))
    fp = np.sum(np.logical_and(labels_pred == 1, labels_test == 2))
    fn = np.sum(np.logical_and(labels_pred == 2, labels_test == 1))
    return tp, tn, fp, fn

def knn_predict(k: int, samples_test: np.array, samples_train: np.array, labels_train: np.array):
    labels_pred = []
    for sample in samples_test:
        distance_and_label = []
        for i in range(len(samples_train)):
            distance = norm(samples_train[i] - sample)
            label = labels_train[i]
            distance_and_label.append((distance, label))
        distance_and_label = np.array(distance_and_label)
        sorted_distance_and_label = distance_and_label[distance_and_label[:, 0].argsort()]
        neighbors = sorted_distance_and_label[:k, 1]
        count_1 = (neighbors == 1).sum()
        count_2 = (neighbors == 2).sum()
        if count_1 > count_2:
            labels_pred.append(1)
        elif count_2 > count_1:
            labels_pred.append(2)
        else:
            print('Beide Klassen möglich')
    labels_pred = np.array(labels_pred)
    return labels_pred

def get_acc_precision_recall_f1(tp: float, tn: float, fp: float, fn: float):
    acc = (tp+tn)/(tp+tn+fp+fn)
    pre = tp/(tp+fp)
    rec = tp/(tp+fn)
    f1 = 2*tp/(2*tp + fp + fn)
    return round(acc, 4), round(pre, 4), round(rec, 4), round(f1, 4)

def show_confusion_matrix(confusion_matrix, title: str):
    ax = sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='g')

    ax.set_title(title)
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values ')

    ax.xaxis.set_ticklabels(['True', 'False'])
    ax.yaxis.set_ticklabels(['True', 'False'])

    # plt.show()

def print_metrics(metrics: dict):
    print('Comparison of own and builtin KNN')
    for elem in metrics:
        print(f'{elem}: -own: {metrics[elem][0]} -builtin: {metrics[elem][1]}')

def print_metrics_cnn(metrics: dict):
    print('Metrics for CNN Classification')
    for elem in metrics:
        print(f'{elem}: {metrics[elem]}')

def load_images_and_labels(image_dir: list, image_type: BCImageEnum):
    image_paths = just_image_names(image_dir)
    images = []
    labels = []

    for path in image_paths[:]:
        img = BCImage(path=path)
        images.append(img[image_type])
        labels.append(img.label)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels

def get_one_hot_encoding(labels: np.array):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse_output=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    labels = onehot_encoder.fit_transform(integer_encoded)
    return labels

def train_nb(x_train, y_train):
    df_train = np.concat([x_train, y_train], axis=1)
    n_features = x_train.shape
    categories = np.unique(df_train['Label'])
    n_samples_pro_category = df_train['Label'].value_counts()
    mean = df_train.groupby('Label').mean()
    variance = df_train.groupby('Label').var(ddof=0)
    return n_features, categories, n_samples_pro_category, mean, variance


if __name__ == '__main__':
    print('Just a bunch of funcitons')