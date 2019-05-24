import numpy as np


class NeuralNetwork:
    # @staticmethod
    # def create(genome: Genome, depth, input, output):
    #     layers = []
    #
    #     x = Input(shape=(input,))
    #     for x in range(0, depth):
    #         x = Dense(10, activation='relu')(x)
    #     all_layers = keras.layers.concatenate(layers)(x)
    #     output = Dense(32)(all_layers)
    #     return x

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def tanh(x):
        return (1 - np.exp(-2 * x)) / (1 + np.exp(-2 * x))

    #TODO: Sometimes layers don't contain output nodes
    @staticmethod
    def feedforward(genome, x_input, y_out):
        graph, rev_graph = genome.create_graphs()
        layers = NeuralNetwork.find_layers(genome)
        connections = genome.list_connections()
        # values = [0 for x in range(0, len(genome.node_genes))]
        # for x in len(x_in):
        #     values[x] = x_in[x]
        error = []
        for xin, yout in list(zip(x_input, y_out)):
            values = {index: xin[index] for index, x in enumerate(xin)}
            all_layers = list(layers)
            for index, layer in enumerate(all_layers[:1]):
                # for layer_node in layer:
                    # get data for whole layers
                data = np.array([values.get(x) for x in layer])
                # get weights for next nodes
                for node in list(all_layers[index+1]):
                    weights = np.array([genome.connection_genes.get(x).weight for x in rev_graph.get(node)])
                    out = np.sum(data.reshape(-1,1) * weights.reshape(1, -1))
                    values[node] = NeuralNetwork.relu(out)

            output = np.array([values.get(x) for x in genome.outputs])
            mse = ((output - np.array(yout)) ** 2).mean()
            error.append(mse)
        return error

    # @staticmethod
    # def compute_node(inputs: np.array, weights: np.array, activation_func):
    #     # data = np.array([values.get(x) for x in rev_graph.get(node)])
    #     # weights = np.array([genome.connection_genes.get(x).weight for x in rev_graph.get(node)])
    #     out = np.sum(data * weights)
    #     values[node] = NeuralNetwork.relu(out)

    #TODO: use reverse graph to generate layer set list
    @staticmethod
    def find_layers(genome):
        # s = set(genome.inputs)
        # layers = [s]
        # connections = genome.list_connections()
        # required = NeuralNetwork.find_required(genome, connections)
        # while 1:
        #     # Find candidate nodes c for the next layer.  These nodes should connect
        #     # a node in s to a node not in s.
        #     c = set(b for (a, b) in connections if a in s and b not in s)
        #     # Keep only the used nodes whose entire input set is contained in s.
        #     t = set()
        #     for n in c:
        #         if n in required and all(a in s for (a, b) in connections if b == n):
        #             t.add(n)
        #     if not t:
        #         break
        #     layers.append(t)
        #     s = s.union(t)
        # return layers
        layers = []
        _, rev_graph = genome.create_graphs

        for node in genome.outputs:
            if rev_graph.get(node):
                print()

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
