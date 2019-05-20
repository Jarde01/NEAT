from keras import Input

from neat import Genome, GenomeFactory, mutate_add_node
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam


class NeuralNetwork:
    @staticmethod
    def create(genome: Genome, depth, input, output):
        layers = []

        x = Input(shape=(input,))
        for x in range(0, depth):
            x = Dense(10, activation='relu')(x)
        all_layers = keras.layers.concatenate(layers)(x)
        output = Dense(32)(all_layers)
        return x


g = GenomeFactory.create_genome(2, 1)
mutate_add_node(g)
nn = NeuralNetwork.create(g, 1, 2, 1)

print()
