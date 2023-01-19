import csv
from scipy.stats import norm
from numpy import mean
from numpy import std
import numpy as np
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

def scale(X, x_min=0, x_max=1):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom

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
            distance = np.linalg.linalg.norm(samples_train[i] - sample)
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
    return labels_pred[0]




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


def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def percentage(data: np.array):
    return data / data.sum()


def fit_distribution(data):
    mu = mean(data)
    sigma = std(data)
    dist = norm(mu, sigma)
    return dist


def get_probabilities(sample: np.array, priories, distributions):
    probabilities = []
    n_features = len(sample)
    probs = []
    for c in distributions:
        dists = distributions[c]
        feature_probs = []
        for i in range(n_features):
            feature_prob = dists[i].pdf(sample[i])
            feature_probs.append(feature_prob)
        probs.append(np.prod(feature_probs))
    for i in range(len(probs)):
        probabilities.append(priories[i]*probs[i])
    return percentage(np.array(probabilities))


def get_priory_prob(labels, samples):
    priory_probs = []
    classes = np.unique(labels)
    for c in classes:
        samples_in_class = samples[labels == c]
        priory_prob = len(samples_in_class) / len(labels)
        priory_probs.append(priory_prob)

    priory_probs = np.array(priory_probs)
    return priory_probs


def get_distributions(labels, samples: np.array):
    feature_distributions = {}
    classes = np.unique(labels)
    for c in classes:
        samples_in_class = samples[labels == c]
        key = f'dists_{c}'
        distributions = []
        for feature in samples_in_class.T:
            distribution = fit_distribution(feature)
            distributions.append(distribution)
        distributions = np.array(distributions)
        feature_distributions[key] = distributions
    return feature_distributions


def fit_naivebayes(labels_train: np.array, samples_train: np.array):
    priories = get_priory_prob(labels=labels_train, samples=samples_train)
    distributions = get_distributions(labels=labels_train, samples=samples_train)
    return priories, distributions


if __name__ == '__main__':
    print('Just a bunch of funcitons')