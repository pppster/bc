import json
from glob import glob
import tensorflow as tf
from bci.utils.bcimageenum import IMAGE, NOBACKGROUND, MASK
from sklearn.model_selection import train_test_split

from functions.functions import *

# paths to the trained models and the training history
models_paths = sorted(glob('.\models\model*'))
history_paths = sorted(glob('.\models\*.json'))

# List of the different classification tasks
class_types = ['label_damaged', 'bottle_damaged', 'open_closed']

# List of the different image types -> find out which image type works best
image_types = [MASK, IMAGE, NOBACKGROUND]

for class_type in class_types[:]:
    for image_type in image_types[:]:

        # get image type as string
        image_type_str = str(image_type)[str(image_type).find('.')+1:]

        # filter the right trained model
        model_path = [model_path for model_path in models_paths if (class_type in model_path and image_type_str in model_path)][0]

        # filter the right history path
        history_path = [history_path for history_path in history_paths if (class_type in history_path and image_type_str in history_path)][0]

        # open history file
        with open(history_path) as f:
            history = json.load(f)

        # load model
        model = tf.keras.models.load_model(model_path)

        # load images and get labels due to directory structure
        img_dir = f'..\images\_{class_type}\*\*'
        images, labels = load_images_and_labels(image_dir=img_dir, image_type=image_type)

        # one hot encoding for labels
        labels = get_one_hot_encoding(labels)

        # Split images and labels in training and test data
        images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.2, random_state=42, shuffle=True)

        # predict labels
        labels_pred = model.predict(images_test)

        # decode one hot encoding for predicted labels and true labels
        labels_pred = np.round(labels_pred)
        labels_pred = np.argmax(labels_pred, axis=1) + 1
        labels_test = np.argmax(labels_test, axis=1) + 1

        # calculate true pos, true-neg, false-pos and false-neg for metric calculation
        tp, tn, fp, fn = get_TP_TN_FP_FN(labels_pred, labels_test)

        # calculate metrics for CNN
        accuracy, precision, recall, f1 = get_acc_precision_recall_f1(tp=tp, tn=tn, fp=fp, fn=fn)

        # store metrics in a dictionary -> easier to store
        metrics = {'Accuracy': accuracy,
                   'Precision': precision,
                   'Recall': recall,
                   'F1-Score': f1}

        # print metrics to console
        print_metrics_cnn(metrics=metrics)

        # calculate confusion matrices for visualization tasks
        confmat = confusion_matrix(labels_test, labels_pred)
        confmat_title = f'Confusion Matrix for {class_type} with {image_type_str} for CNN'
        show_confusion_matrix(confusion_matrix=confmat, title=confmat_title)
        plt.show()

        # plot training history for validation and training data
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
