import tensorflow as tf
import numpy as np
import random
import enum
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

innovation_number = 0

# https://github.com/CodeReclaimers/neat-python/blob/master/examples/xor/evolve-minimal.py
# 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]


def model_crossover(model1, model2):
    w1 = models[model1].get_weights()
    w2 = models[model2].get_weights()
    neww1 = w1
    neww2 = w2
    neww1[0] = w2[0]
    neww2[0] = w1[0]
    return np.asarray([neww1, neww2])


def mutate(weights):
    for xi in range(len(weights)):
        for yi in range(len(weights[xi])):
            if random.uniform(0,1) > 0.85:
                change = random.uniform(-0.5, 0.5)
                weights[xi][yi] += change
    return weights


def predict(input, model_num):
    output_prob = models[model_num].predict(input, 1)[0]
    print(output_prob)


models = []
num_models = 5
for i in range(num_models):
    model = Sequential()
    model.add(Dense(output_dim=7, input_dim=2))
    model.add(Activation('sigmoid'))
    model.add(Dense(output_dim=2))
    model.add(Activation("sigmoid"))

predict(xor_inputs)