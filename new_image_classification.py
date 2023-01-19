from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from SVM.svm_eval import SVM
from bci.core.bcimage import BCImage
from bcdf.core.bcdataframe import BCDataframe
from bcdf.utils.utils import *
import pandas as pd
import numpy as np

from functions.functions import *
import tensorflow as tf

####################
new_image_paths = ['.\\image_to_classify\\damaged_00014.jpg',
                   '.\\image_to_classify\\intact_00008.jpeg',
                   '.\\image_to_classify\\open_00014.jpg']
new_image_dir = '.\\image_to_classify'
####################

k=3

df_unnorm_paths = [
    '.\\image_to_classify\\unnormalized_dataframes\\bottle_damaged_dataframe.csv',
    '.\\image_to_classify\\unnormalized_dataframes\\label_damaged_dataframe.csv',
    '.\\image_to_classify\\unnormalized_dataframes\\open_closed_dataframe.csv'
    ]

df_norm_paths = [
    '.\\dataframes\\_bottle_damaged\\dataframe.csv',
    '.\\dataframes\\_label_damaged\\dataframe.csv',
    '.\\dataframes\\_open_closed\\dataframe.csv'
]
img_dirs = [
    '.\\images\\_bottle_damaged\\*\\*',
    '.\\images\\_label_damaged\\*\\*',
    '.\\images\\_open_closed\\*\\*'
    ]

cnn_paths = [
    '.\\CNN\\models\\model_MASK_bottle_damaged',
    '.\\CNN\\models\\model_MASK_label_damaged',
    '.\\CNN\\models\\model_MASK_open_closed',
]

classification_tasks = [
    'bottle_damaged',
    'label_damaged',
    'open_closed'
]

label_encoding = {
    'bottle_damaged': {
        1: 'damaged',
        2: 'intact'
    },
    'label_damaged': {
        1: 'damaged',
        2: 'intact'
    },
    'open_closed': {
        1: 'closed',
        2: 'open'
    }
}


# preprocessing
for new_image_path in new_image_paths:
    image_to_classify = BCImage(path=new_image_path)
    image_to_classify.load_image()
    image_to_classify.resize_image()
    image_to_classify.padding_image()
    image_to_classify.remove_background()
    image_to_classify.generate_mask()
    image_to_classify.save_mask()
    image_to_classify.save_no_background()
    image_to_classify.save_image()
    img = BCImage(path=new_image_path)
    area = get_area(img)
    extend = get_extend(img)
    perimeter = get_perimeter(img)
    solidity = get_solidity(img)
    centroid_x = get_centroid_x(img)
    num_corners = get_num_corners(img)
    aspect_ratio = get_aspect_ratio(img)
    new_img_features = np.array([
        aspect_ratio,
        num_corners,
        centroid_x,
        solidity,
        perimeter,
        extend,
        area
        ], dtype='float')
    im = img.mask


    for classification_task in classification_tasks:
        df_unnorm_path = [df_unnorm_path for df_unnorm_path in df_unnorm_paths if classification_task in df_unnorm_path][0]
        df_unnorm = read_csv_as_np(df_unnorm_path)
        df_unnorm = df_unnorm[1:, 1:-1].astype('float')
        max_of_col = np.amax(df_unnorm, axis=0)
        min_of_col = np.amin(df_unnorm, axis=0)
        sample = (new_img_features-min_of_col)/(max_of_col-min_of_col)

        # load trainings df
        df_norm_path = [df_norm_path for df_norm_path in df_norm_paths if classification_task in df_norm_path][0]
        df_norm = read_csv_as_np(path=df_norm_path)
        labels, samples = get_labels_and_samples(df_norm)
        samples_train, samples_test, labels_train, labels_test = train_test_split(samples,
                                                                                  labels,
                                                                                  test_size=0.2,
                                                                                  random_state=42,
                                                                                  shuffle=True)


        # load CNNs
        model_path = [model_path for model_path in cnn_paths if classification_task in model_path][0]
        model = tf.keras.models.load_model(model_path)
        im = im.reshape((1, 300, 225, 1))
        labels_pred_own_cnn = model.predict(im)
        probabilities_cnn = labels_pred_own_cnn[0]
        labels_pred_own_cnn = np.round(np.array(probabilities_cnn))
        labels_pred_own_cnn = np.argmax(labels_pred_own_cnn) + 1

        # fit GNB
        gnb = GaussianNB()
        gnb.fit(X=samples_train, y=labels_train)
        labels_pred_builtin_gnb = gnb.predict([sample])

        priories, distributions = fit_naivebayes(labels_train=labels_train, samples_train=samples_train)
        # predict GNB
        probabilities = get_probabilities(sample=sample, priories=priories, distributions=distributions)
        # print(probabilities)

        # encode labels for comparison
        labels_pred_own_gnb = np.round(np.array(probabilities))
        labels_pred_own_gnb = np.argmax(probabilities) + 1

        labels_pred_own_knn = knn_predict(k=k,
                                          samples_train=samples_train,
                                          samples_test=[sample],
                                          labels_train=labels_train)
        knn_builtin = KNeighborsClassifier(n_neighbors=k)
        knn_builtin.fit(samples_train, labels_train)
        labels_pred_builtin_knn = knn_builtin.predict([sample])

        svm_own = SVM()
        svm_builtin = SVC()
        labels_train_svm = np.where(labels_train == 1, 1, -1)
        svm_own.load_model(path=f'.\\SVM\\models\\{classification_task}_model')
        svm_builtin.fit(samples_train, labels_train_svm)
        #
        labels_pred_own_svm = svm_own.predict(sample)
        labels_pred_builtin_svm = svm_builtin.predict([sample])
        #
        labels_pred_builtin_svm = np.where(labels_pred_builtin_svm == 1, 1, 2)
        labels_pred_own_svm = np.where(labels_pred_own_svm == 1, 1, 2)
        print(labels_pred_own_svm)

        print(f'classification for: {classification_task}')
        print(f'Predicted label of the self-built CNN: {label_encoding[classification_task][labels_pred_own_cnn]}')
        print(f'Predicted label of the self-built GNB: {label_encoding[classification_task][labels_pred_own_gnb]}')
        print(f'Predicted label of the built-in GNB: {label_encoding[classification_task][labels_pred_builtin_gnb[0]]}')
        print(f'Predicted label of the self-built KNN: {label_encoding[classification_task][labels_pred_own_knn]}')
        print(f'Predicted label of the built-in KNN: {label_encoding[classification_task][labels_pred_builtin_knn[0]]}')
        print(f'Predicted label of the self-built SVM: {label_encoding[classification_task][int(labels_pred_own_svm)]}')
        print(f'Predicted label of the built-in SVM: {label_encoding[classification_task][labels_pred_builtin_svm[0]]}')

