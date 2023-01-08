import json
from glob import glob
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from bci.utils.bcimageenum import IMAGE, NOBACKGROUND, MASK
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from functions.functions import *

models_paths = sorted(glob('.\models\model*'))
history_paths = sorted(glob('.\models\*.json'))
class_types = ['label_damaged', 'bottle_damaged', 'open_closed']
image_types = [MASK, IMAGE, NOBACKGROUND]

for class_type in class_types[:]:
    for image_type in image_types[:]:
        image_type_str = str(image_type)[str(image_type).find('.')+1:]
        model_path = [model_path for model_path in models_paths if (class_type in model_path and image_type_str in model_path)][0]
        history_path = [history_path for history_path in history_paths if (class_type in history_path and image_type_str in history_path)][0]
        with open(history_path) as f:
            history = json.load(f)
        model = tf.keras.models.load_model(model_path)

        img_dir = f'..\images\_{class_type}\*\*'
        images, labels = load_images_and_labels(image_dir=img_dir, image_type=image_type)
        labels = get_one_hot_encoding(labels)

        num_categories = labels.shape[1]

        images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.2, random_state=42, shuffle=True)

        labels_pred = model.predict(images_test)

        labels_pred = np.round(labels_pred)
        labels_pred = np.argmax(labels_pred, axis=1) + 1
        labels_test = np.argmax(labels_test, axis=1) + 1

        tp, tn, fp, fn = get_TP_TN_FP_FN(labels_pred, labels_test)
        accuracy, precision, recall, f1 = get_acc_precision_recall_f1(tp=tp, tn=tn, fp=fp, fn=fn)
        metrics = {'Accuracy': accuracy,
                   'Precision': precision,
                   'Recall': recall,
                   'F1-Score': f1}
        print(f'{image_type_str} - {class_type}')
        print_metrics_cnn(metrics=metrics)
        confmat = confusion_matrix(labels_test, labels_pred)
        confmat_title = f'Confusion Matrix for {class_type} with {image_type_str} for CNN'
        show_confusion_matrix(confusion_matrix=confmat, title=confmat_title)
        plt.show()

        plt.plot(history['val_loss'])
        plt.plot(history['val_acc'])
        plt.title(f'Accuracy and Loss for {class_type} with {image_type_str} for validation data')
        plt.xlabel('Epochs')
        plt.legend(['Loss', 'Accuracy'])
        plt.show()

        plt.plot(history['loss'])
        plt.plot(history['acc'])
        plt.title(f'Accuracy and Loss for {class_type} with {image_type_str} for training data')
        plt.xlabel('Epochs')
        plt.legend(['Loss', 'Accuracy'])
        plt.show()


# false_counter = 0
# for i in range(len(labels_pred)):
#     comp = np.argmax(labels_pred[i]) == np.argmax(labels_test[i])
#     print(f'{comp}')
#     if not comp:
#         false_counter += 1
# f1 = 2*history['val_precision'][-1]*history['val_recall'][-1]/(history['val_precision'][-1]+history['val_recall'][-1])
#
# print(f'From {len(labels_pred)} test-samples are {false_counter} wrong classified. Accuracy: {(len(labels_pred) - false_counter) / len(labels_pred) *100 }%')
# print(f'F1-Score: {f1}')
#
#
# plt.plot(history['val_categorical_accuracy'])
# plt.plot(history['val_categorical_crossentropy'])
# plt.title('Accuracy und Loss der Validierungsdaten')
# plt.xlabel('Epochs')
# plt.legend(['Accuracy', 'Loss'])
# plt.show()
#
#

#
# TP = np.sum(np.logical_and(labels_pred == 0, labels_test == 0))
# TN = np.sum(np.logical_and(labels_pred == 1, labels_test == 1))
# FP = np.sum(np.logical_and(labels_pred == 0, labels_test == 1))
# FN = np.sum(np.logical_and(labels_pred == 1, labels_test == 0))
#
# print(f'Summe: {TP + TN + FP + FN}, TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}')
# cf_matrix = confusion_matrix(labels_test, labels_pred)
# print(cf_matrix)
# pre = TP / (TP + FP)
# rec = TP / (TP + FN)
# f1 = 2 * TP / (2 * TP + FP + FN)
#
# print(f'Recall: {rec}')
# print(f'Precision: {pre}')
# print(f'F1-score: {f1}')
#
# ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='g')
#
# ax.set_title(f'Confusion Matrix Sicherungsetikett intakt')
# ax.set_xlabel('\nPredicted Values')
# ax.set_ylabel('Actual Values ');
#
# ## Ticket labels - List must be in alphabetical order
# ax.xaxis.set_ticklabels(['False', 'True'])
# ax.yaxis.set_ticklabels(['False', 'True'])
#
# ## Display the visualization of the Confusion Matrix.
# plt.show()
