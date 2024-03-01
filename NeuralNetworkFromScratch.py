import math
import random
import numpy as np
import sympy as sm
import tensorflow as tf
from keras.src.datasets import mnist
'''
data=tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#Normalize the values between 0 and 0
train_images, test_images = train_images / 255.0, test_images / 255.0
'''

class neuron:
    input = None
    weight = None
    expected = None

    def __init__(self):
        self.input = random.random()
        self.weight = 0.5
        self.expected = 2


class layer:
    neurons = []


def calculate(neuron1):
    return neuron1.input * neuron1.weight


def sigmoid(result):
    return 1 / (1 + math.exp(-result))


##cross entropy loss function is used for classification problems.
def cross_entropy_loss(target_value, predicted_value):
    loss = - (target_value * np.log(predicted_value) + (1 - target_value) * np.log(1 - predicted_value)).mean()
    return loss


def backpropagation(i, w, y):
    print()

    a = i * w
    q = 1 / (1 + sm.exp(a))
    l = (q - y) ** 2

    dl = 2 * (q - y)
    dq = (1 / 1 + sm.exp(-a) * 1 - 1 / 1 + sm.exp(-a))
    da = i

    result = dl * dq * da
    return result


def gradient_descend(value, learning_rate):
    print(f"Previous Weight= {n1.weight}")
    n1.weight = n1.weight - learning_rate * value
    print(f"Updated Weight= {n1.weight}")


def early_stopping(loss_values, threshold=0.001, patience=5):
    if len(loss_values) > patience:
        recent_losses = loss_values[-patience:]
        if max(recent_losses) - min(recent_losses) < threshold:
            return True
    return False


n1 = neuron()
loss_values = []

# Input Layer
for epoch in range(10000):
    n1_calculation = calculate(n1)

    output = n1_calculation
    sigmoid_output = sigmoid(output)

    loss_value = cross_entropy_loss(2, sigmoid_output)
    print(f"loss_value {loss_value}")

    loss_values.append(loss_value)

    print(f"Epoch {epoch + 1}, Loss: {loss_value}")

    gradient_descend((backpropagation(n1.input, n1.weight, n1.expected).evalf()), 0.1)
    # Check for early stopping
    if early_stopping(loss_values):
        print("Stopping early.")
        break
print(f"Last Weight {n1.weight}")
