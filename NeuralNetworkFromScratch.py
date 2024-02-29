import math
import random
import numpy as np

train_data=[[1,2,3]]
test_data=[[1,2,3]]
class neuron:
    input = None
    weight = None

    def __init__(self):
        self.input = random.random()
        self.weight = 0.5


class layer:
    neurons = []


def calculate(neuron1):
    return neuron1.input * neuron1.weight


def sigmoid(result):
    return 1 / (1 + math.exp(-result))


def loss_calculation(target_value, predicted_value):
    loss = - (target_value * np.log(predicted_value) + (1 - target_value) * np.log(1 - predicted_value)).mean()
    return loss


n1 = neuron()
n2 = neuron()
# Input Layer
input_layer = layer()
input_layer.neurons.append(n1)

c = calculate(n1, n2)
print(c)
print(sigmoid(c))
print(loss_calculation(2, sigmoid(c)))
np.gradient(loss_calculation())


