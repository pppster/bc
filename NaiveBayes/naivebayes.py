from sklearn.model_selection import train_test_split
from functions.functions import *
from scipy.stats import norm
from numpy import mean
from numpy import std

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


path_df_open_closed = '../dataframes/_open_closed/dataframe.csv'
df_open_closed = read_csv_as_np(path_df_open_closed)
labels, samples = get_labels_and_samples(df_open_closed)
samples_train, samples_test, labels_train, labels_test = train_test_split(samples,
                                                                          labels,
                                                                          test_size=0.2,
                                                                          random_state=42,
                                                                          shuffle=True)

priories, distributions = fit_naivebayes(labels_train=labels_train, samples_train=samples_train)
labels_pred = []
for sample in samples_test:
    probabilities = get_probabilities(sample=sample, priories=priories, distributions=distributions)
    labels_pred.append(probabilities)
labels_pred = np.round(np.array(labels_pred))
labels_pred = np.argmax(labels_pred, axis=1)
labels_test = get_one_hot_encoding(labels_test)
labels_test = np.argmax(labels_test, axis=1)
right_classified = (labels_pred == labels_test).sum()
n_samples = len(labels_pred)
print(f'From {n_samples} samples are {right_classified} right classified. Acc: {right_classified/n_samples}')



# # generate 2d classification dataset
# X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
# # sort data into classes
# Xy0 = X[y == 0]
# Xy1 = X[y == 1]
# # calculate priors
# priory0 = len(Xy0) / len(X)
# priory1 = len(Xy1) / len(X)
# # create PDFs for y==0
# distX1y0 = fit_distribution(Xy0[:, 0])
# distX2y0 = fit_distribution(Xy0[:, 1])
# # create PDFs for y==1
# distX1y1 = fit_distribution(Xy1[:, 0])
# distX2y1 = fit_distribution(Xy1[:, 1])
# # classify one example
# Xsample, ysample = X[0], y[0]
# py0 = probability(Xsample, priory0, distX1y0, distX2y0)
# py1 = probability(Xsample, priory1, distX1y1, distX2y1)
# print('P(y=0 | %s) = %.3f' % (Xsample, py0 * 100))
# print('P(y=1 | %s) = %.3f' % (Xsample, py1 * 100))
# print('Truth: y=%d' % ysample)