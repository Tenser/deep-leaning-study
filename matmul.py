import tensorflow as tf
from tensorflow.keras import datasets, layers, Model
import numpy as np

class MyLayer(layers.Layer):
    def __init__(self, output_size):
        self.output_size = output_size
        super(MyLayer, self).__init__()

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[1], self.output_size), initializer='random_normal', trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.matmul1 = MyLayer(100)
        self.avg1 = layers.Average()
        self.matmul2 = MyLayer(10)

    def call(self, x, training=False):
        a1 = self.matmul1(x)
        a2 = self.matmul1(x)
        a3 = self.avg1([a1, a2])
        return self.matmul2(a3)

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

train_images = train_images.reshape((60000, -1))
test_images = test_images.reshape((10000, -1))

model = MyModel()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=3, batch_size=100)

test_loss, test_acc = model.evaluate(test_images, test_labels)