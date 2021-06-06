import tensorflow as tf
from tensorflow.keras import Model, Sequential, Input


# Multiple outputs
# MNIST 에 대한 간단한 오토인코더 모델을 만들어보겠습니다.

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


