from keras import Input
import numpy as np
from neat import Genome, GenomeFactory, mutate_add_node, list_connections, create_network
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

    @staticmethod
    def feedforward(genome: Genome, x_input, y_out):
        output = None
        graph, rev_graph = create_network(genome)
        layers = NeuralNetwork.find_layers(genome)
        values = [0 for x in range(0, len(genome.node_genes))]
        # for x in len(x_in):
        #     values[x] = x_in[x]
        error = []
        for xin, yout in list(zip(x_input, y_out)):
            for layer in list(layers):
                for node in layer:
                    data = np.array(xin)
                    weights = np.array([genome.connection_genes.get(x).weight for x in rev_graph.get(node)])
                    values[node] = np.sum(data*np.array([genome.connection_genes.get(x).weight for x in rev_graph.get(node)]))
            output = np.array(values[len(genome.inputs): len(genome.inputs) + len(genome.outputs)])
            mse = ((output - yout) ** 2).mean(axis=None)
            error.append(mse)
        return error

    @staticmethod
    def find_layers(genome: Genome):
        layers = []
        s = set(genome.inputs)
        connections = list_connections(genome)
        required = NeuralNetwork.find_required(genome, connections)

        while 1:
            # Find candidate nodes c for the next layer.  These nodes should connect
            # a node in s to a node not in s.
            c = set(b for (a, b) in connections if a in s and b not in s)
            # Keep only the used nodes whose entire input set is contained in s.
            t = set()
            for n in c:
                if n in required and all(a in s for (a, b) in connections if b == n):
                    t.add(n)

            if not t:
                break

            layers.append(t)
            s = s.union(t)

        return layers

    @staticmethod
    def find_required(genome, connections):
        inputs = genome.inputs
        required = set(genome.outputs)
        s = set(genome.outputs)
        while 1:
            # Find nodes not in S whose output is consumed by a node in s.
            t = set(a for (a, b) in connections if b in s and a not in s)

            if not t:
                break

            layer_nodes = set(x for x in t if x not in inputs)
            if not layer_nodes:
                break

            required = required.union(layer_nodes)
            s = s.union(t)

        return required
