from datetime import datetime
import numpy as np
import json
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from bci.utils.bcimageenum import IMAGE, MASK, NOBACKGROUND, BCImageEnum
from bci.core.bcimage import BCImage
from bcdf.utils.utils import just_image_names
class_types = [
    'open_closed']
image_types = [NOBACKGROUND]
batch_size = 32
epochs = 20
kernel_size = (3, 3)
activation = 'relu'
feature_maps = 16
fully_connected = 100

for image_type in image_types:
    for class_type in class_types:
        i = str(image_type)[str(image_type).find('.')+1:]

        img_dir = f'..\images\_{class_type}\*\*'
        image_paths = just_image_names(img_dir)
        images = []
        labels = []

        for path in image_paths[:]:
            img = BCImage(path=path)
            images.append(img[image_type])
            labels.append(img.label)

        input_shape = images[0].shape

        images = np.array(images)
        labels = np.array(labels)

        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(labels)
        onehot_encoder = OneHotEncoder(sparse_output=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        labels = onehot_encoder.fit_transform(integer_encoded)
        num_categories = labels.shape[1]

        print(f'Labels: {labels.shape} --- Images: {images.shape}')
        images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.2, random_state=42, shuffle=True)
        if image_type == MASK:
            images_train = images_train.reshape((images_train.shape[0], 300, 225, 1))
            input_shape = images_train[0].shape
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(feature_maps, kernel_size=kernel_size, activation=activation, input_shape=input_shape),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(2*feature_maps, kernel_size=kernel_size, activation=activation),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(4*feature_maps, kernel_size=kernel_size, activation=activation),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(8*feature_maps, kernel_size=kernel_size, activation=activation),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(5*fully_connected, activation=activation),
            tf.keras.layers.Dropout(0.1, seed=2019),
            tf.keras.layers.Dense(4*fully_connected, activation=activation),
            tf.keras.layers.Dropout(0.3, seed=2019),
            tf.keras.layers.Dense(3*fully_connected, activation=activation),
            tf.keras.layers.Dropout(0.4, seed=2019),
            tf.keras.layers.Dense(2*fully_connected, activation=activation),
            tf.keras.layers.Dropout(0.2, seed=42),
            tf.keras.layers.Dense(fully_connected, activation=activation),
            tf.keras.layers.Dense(num_categories, activation="softmax")
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        history = model.fit(x=images_train, y=labels_train, validation_split=0.1, batch_size=batch_size, epochs=epochs)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = Path(f'cnn\{timestamp}_{i}_{class_type}')
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename_json = Path(f'{filename}.json')
        filename_json.parent.mkdir(parents=True, exist_ok=True)
        model.save(filename)
        with open(filename_json, 'w') as fp:
            json.dump(history.history, fp)


