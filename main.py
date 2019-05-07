import tensorflow as tf
from enum import Enum
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

innovation_num = 1

xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]


class NodeType(Enum):
    Input = 1
    Hidden = 2
    Output = 3


class GeneType(Enum):
    Sensor = 1
    Output = 2
    Hidden = 3


class NodeGene:
    def __init__(self, nodenum, nodetype):
        self.node_number = nodenum
        self.node_type = nodetype


class Connection:
    def __init__(self):
        global innovation_num
        self.in_node_key = 0
        self.out_node_key = 0
        self.weight = 0.0
        self.enabled = True
        self.innovation_num = 0  # help to find corresponding genes


class Population:
    def __init__(self):
        self.Genomes = []

    def add(self, genome):
        self.Genomes.append(genome)


class Genome:
    def __init__(self):
        self.key = 1
        self.fitness = 0
        self.connection_genes = {}
        self.node_genes = {}
        # self.ancestors = {}

    def create_graph(self):
        for key, node in self.connection_genes.items():
            print(node)

    def run(self, x, y):
        # create graph
        graph = self.create_graph()

        print("test")
        # # find the input nodes
        # for input in zip(x, y):
        #     for index, val_tuple in enumerate(input):
        #         x, y = val_tuple
        #         next = self.connection_genes[index].out_node_key
        #         while next is not None:
        #             result = self.connection_genes[index].pass_value(x)
        #         pass

        pass

    def mutate(self):
        pass


class GenomeFactory:
    @staticmethod
    def create_genome(num_input_nodes, num_output_nodes):
        global innovation_num
        genome = Genome()

        curr_gene_num = 1
        input_genes = []
        for x in range(num_input_nodes):
            n = NodeGene(nodenum=curr_gene_num, nodetype=NodeType.Input)
            genome.node_genes[curr_gene_num] = n
            input_genes.append(n)
            curr_gene_num += 1

        output_genes = []
        for x in range(num_output_nodes):
            n = NodeGene(nodenum=curr_gene_num, nodetype=NodeType.Output)
            output_genes.append(n)
            genome.node_genes[curr_gene_num] = n
            curr_gene_num += 1

        # connect all the input to the output genes
        node_count = 1
        for in_gene in input_genes:
            for j, out_gene in enumerate(output_genes):
                e = Connection()
                e.in_node_key = in_gene.node_number
                e.out_node_key = out_gene.node_number
                genome.connection_genes[node_count] = e
                node_count += 1
        return genome


def speciation(genome1, genome2):
    N = max([len(genome1), len(genome2)])
    E = 1  # number of excess genes
    D = 1  # number of disjoint genes
    W = 0  # weight difference of matching genes
    c1 = 1  # coefficients to adjust importance of certain factors
    c2 = 1
    c3 = 1

    distance = c1 * E / N + c2 * D / N + c3 * W


# Best performing r% of each species is randomply mated to generate Nji offspring,
# replacing the entire population of the species
def fitness():
    Nj = None  # num of old individuals
    Nji = None  # num of individuals in species j
    fij = None  # adjusted fitness of individual i in species j
    f = 1  # mean adjusted fitness in the entire population
    Nji = sum(Nj * fij) / f
    pass


# add a new connection with a random weight to two previously unconnected nodes
# connection to new node is 1, from new node to forward node is same as current weight
def mutate_add_connection(node1, node2):
    pass


# split existing connection and place new node in between
def mutate_add_node(node1, node2):
    pass


def create_with_hidden_layer(genome2):
    global innovation_num
    new_node_key = 4
    new_out_node_key = 3
    conn_modifying_key = 2

    genome2.node_genes[new_node_key] = NodeGene(new_node_key, NodeType.Hidden)
    newcon = Connection()
    newcon.innovation_num = innovation_num
    newcon.in_node_key = new_node_key
    newcon.out_node_key = new_out_node_key
    genome2.connection_genes[new_out_node_key] = newcon
    genome2.connection_genes[conn_modifying_key].out_node_key = new_node_key

    return genome2


def create_disjoint_genomes():
    genome1 = GenomeFactory().create_genome(2, 1)

    genome2 = GenomeFactory().create_genome(2, 1)
    genome2_hid = create_with_hidden_layer(genome2)
    return genome1, genome2_hid


def crossover(genome1, genome2):
    print()


def test_crossover():
    crossover(*create_disjoint_genomes())


test_crossover()


def test_create_genome():
    tests = [(1, 1), (2, 1), (9, 9)]
    for num_i, num_o in tests:
        g1 = GenomeFactory.create_genome(num_i, num_o)
        connection_count = len(g1.connection_genes)

        print(f'Expected: {num_i * num_o}, Found: {connection_count}')
        assert (connection_count == num_i * num_o)


def test():
    out = Connection()
    out.out_node_key = None

    c1 = Connection()
    c1.in_node_key = 1
    c1.out_node_key = 3

    c2 = Connection()
    c2.in_node_key = 2
    c2.out_node_key = 3

    n1 = NodeGene(1, NodeType.Input)
    n2 = NodeGene(2, NodeType.Input)
    n3 = NodeGene(3, NodeType.Output)

    gene = Genome()
    gene.node_genes = {1: n1, 2: n2, 3: n3}
    gene.connection_genes = {1: c1, 2: c2, 3: out}
