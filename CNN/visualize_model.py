import tensorflow as tf
import visualkeras

model_path = './models/model_IMAGE_open_closed'

model = tf.keras.models.load_model(model_path)

fname = '/model_1.png'
print(model.summary())
visualkeras.layered_view(model=model)
