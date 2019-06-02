import numpy as np


class NeuralNetwork:
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
    def feedforward(genome, x_input, y_out, fitness_fnc):
        graph, rev_graph = genome.create_graphs()
        layers = NeuralNetwork.find_layers(genome)
        # connections = genome.list_connections()
        # values = [0 for x in range(0, len(genome.node_genes))]
        # for x in len(x_in):
        #     values[x] = x_in[x]
        error = []
        for xin, yout in list(zip(x_input, y_out)):
            values = {index: xin[index] for index, x in enumerate(xin)}
            all_layers = list(layers)
            for index, layer in enumerate(all_layers[:-1]):
                data = np.array([values.get(x) for x in layer]).reshape(1, -1)
                # get weights for next nodes
                next_layer = all_layers[index+1]
                for node in list(next_layer):
                    conns = rev_graph.get(node, [])
                    conn_nodes = [genome.find_connection(in_node=x, out_node=node) for x in conns]
                    temp_weights = [x.weight if x is not None else 0 for x in conn_nodes]
                    weights = np.array(temp_weights).reshape(-1, 1)
                    out = np.sum(data * weights)
                    values[node] = NeuralNetwork.relu(out)

            output = np.array([values.get(x) for x in genome.outputs])
            mse = ((output - np.array(yout)) ** 2).mean()
            # print(mse)
            error.append(mse)
        genome.fitness = fitness_fnc(error)
        return error


    #TODO: use reverse graph to generate layer set list
    # BUG: infinite looping when finding layers (1 was connected to itself)
    @staticmethod
    def find_layers(genome):
        layers = []
        _, rev_graph = genome.create_graphs()

        curr_layer = genome.outputs
        layers.insert(0, set(curr_layer))
        layer_count = 0
        while len(curr_layer) > 0:
            print(f"curr_layer: {layer_count}")
            curr = []
            # for all nodes in current layer, get the node above in the graph
            for node in curr_layer:
                curr.extend(rev_graph.get(node)) if rev_graph.get(node, None) is not None else curr
            layers.insert(0, set(curr)) if len(curr) > 0 else layers
            curr_layer = curr
            layer_count += 1
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
