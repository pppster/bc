from pprint import pprint
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from bci.core.bcimage import BCImage
from bci.utils.bcimageenum import IMAGE
from bcdf.utils.utils import just_image_names

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
img_dir = '../../images/_open_closed/*/*'
image_paths = just_image_names(img_dir)
images = []
labels = []

for path in image_paths[:]:
    img = BCImage(path=path)
    images.append(img[IMAGE])
    labels.append(img.label)

images = np.array(images)
labels = np.array(labels)

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)
print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse_output=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
labels = onehot_encoder.fit_transform(integer_encoded)
print(labels.shape)
num_categories = labels.shape[1]


print(f'Labels: {labels.shape} --- Images: {images.shape}')
images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.2, random_state=42, shuffle=True)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=(300, 225, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(550, activation="relu"),
    tf.keras.layers.Dropout(0.1, seed=2019),
    tf.keras.layers.Dense(400, activation="relu"),
    tf.keras.layers.Dropout(0.3, seed=2019),
    tf.keras.layers.Dense(300, activation="relu"),
    tf.keras.layers.Dropout(0.4, seed=2019),
    tf.keras.layers.Dense(200, activation="relu"),
    tf.keras.layers.Dropout(0.2, seed=42),
    tf.keras.layers.Dense(num_categories, activation="softmax")
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.fit(x=images_train, y=labels_train, batch_size=64, epochs=5)
labels_pred = model.predict(images_test)

pprint(labels_pred)
#
#
#
#
#
