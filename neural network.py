import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = tf.keras.datasets.fashion_mnist
#load_data returns us  four numpy arrays. we basically split data
(train_images, train_labels), (test_images, test_labels) = data.load_data()
#our labels from 0 to 9. we create an array for corresponding string.
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# train images is an 3d array. it has 60000 different 2 d array that each of them represent different picture. We plot 63th image inside train images below
plt.imshow(train_images[63])
plt.show()
#preprocessing the data
train_images = train_images/255.0
test_images = test_images/255.0


model=tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(28,28)),#784 input(first) layers
  tf.keras.layers.Dense(128,activation="relu"),# 128 hidden layers. activation funtion is rectified linear unit. outputs are between 0-1
  tf.keras.layers.Dense(10,activation="softmax")#10 output layers. each layers correspond a diifferent image shape



 ]
)
#Training the Model
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit(train_images,train_labels,epochs=5)

#Testing the Model
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('\nTest accuracy:', test_acc)
