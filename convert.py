import tensorflow as tf
from tensorflow import keras


model = keras.models.load_model('mobilenet_final')


converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with tf.io.gfile.GFile('mobilet_flower_v3.tflite', 'wb') as f:
    f.write(tflite_model)





