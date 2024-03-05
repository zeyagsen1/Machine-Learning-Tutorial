import math
import random
import numpy as np
import sympy as sm

# import tensorflow as tf
# from keras.src.datasets import mnist
'''
data=tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#Normalize the values between 0 and 0
train_images, test_images = train_images / 255.0, test_images / 255.0
'''


class neuron:
    input = None
    weight = None
    name = None

    def __init__(self):
        self.input = random.random()
        self.weight = random.random()


class layer:
    neurons = []


def weighted_sum(neuron1, neuron2):
    return neuron1.input * neuron1.weight + neuron2.input * neuron2.weight


def sigmoid(result):
    return 1 / (1 + math.exp(-result))


## cross entropy loss function is used for classification problems.
def cross_entropy_loss(target_value, predicted_value):
    loss = - (target_value * np.log(predicted_value) + (1 - target_value) * np.log(1 - predicted_value)).mean()
    return loss


def mean_squared_error(actual_value, predicted_value):
    # Loop through each sample
    # Calculate squared differences for each sample
    squared_diff_sample = np.square(predicted_value - actual_value)

    return squared_diff_sample


def backpropagation(i1, y):
    print()

    a = n1.input * n1.weight + n2.input + n2.weight
    q = 1 / (1 + sm.exp(-a))
    l = (q - y) ** 2

    dl = 2 * (q - y)
    dq = (q * (1 - q))
    da = i1

    result = dl * dq * da
    return result


def gradient_descend(corresponding_neuron, value, learning_rate):
    print(f"Previous Weight{corresponding_neuron.name}= {corresponding_neuron.weight}")

    corresponding_neuron.weight = corresponding_neuron.weight - learning_rate * value

    print(f"Updated Weight{corresponding_neuron.name}= {corresponding_neuron.weight}")


n1 = neuron()
loss_values = []

n2 = neuron()
#
x_train = np.array([[1, 2],
                    [0, 1]])

y_train = np.array([3, 5]
                   )

test = [2, 3]

output_neuron = neuron()

n1.input = x_train[0][0]
n1.name = "N1"
n2.input = x_train[0][1]
n2.name = "N2"


def train():
    for epoch in range(10000):

        n_calculation = weighted_sum(n1, n2)

        output = n_calculation
        sigmoid_output = sigmoid(output)
        output_neuron.input = sigmoid_output
        loss_value = mean_squared_error(y_train[0], sigmoid_output)

        print(f"Prediction= {output_neuron.input}")

        loss_values.append(loss_value)

        print(f"Epoch {epoch + 1}, Loss: {loss_value}")

        learning_coefficient_n1 = (backpropagation(n1.input, y_train[0]))

        learning_coefficient_n2 = (backpropagation(n2.input, y_train[0]))

        print(f"VALUES n1 {learning_coefficient_n1} n2 {learning_coefficient_n2}")

        gradient_descend(n1, learning_coefficient_n1, 0.6)
        gradient_descend(n2, learning_coefficient_n2, 0.6)

        # Check for early stopping
        if loss_value < 2:
            print(f"loss_value {loss_value}")
            print("TRAINING END!")
            break


train()



''''
n1.input = test[0]
n2.input = test[1]
n_calculation = weighted_sum(n1, n2)
sigmoid_output = sigmoid(n_calculation)
loss_value = mean_squared_error(y_train, np.array([[sigmoid_output, sigmoid_output]]))

print(f"Test Loss: {loss_value}")
print(f"Predicted Output: {sigmoid_output}")
print(f"Weights: {n1.weight}, {n2.weight}")
'''
