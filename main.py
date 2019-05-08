import tensorflow as tf
from enum import Enum
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from collections import OrderedDict
import configparser

config = configparser.ConfigParser()
config.read("config.ini")

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


class ConnectionGene:
    def __init__(self):
        global innovation_num
        self.in_node_key = 0
        self.out_node_key = 0
        self.weight = 0.0
        self.enabled = True
        self.innovation_num = 0  # help to find corresponding genes

    def copy(self):
        new = ConnectionGene()
        new.in_node_key = self.in_node_key
        new.out_node_key = self.out_node_key
        new.weight = self.weight
        new.enabled = self.enabled
        new.innovation_num = self.innovation_num

    def randomize_weight(self):
        self.weight = random.uniform(0, 1)


class Population:
    def __init__(self):
        self.species = OrderedDict()


def update_innovation_number():
    global innovation_num
    innovation_num += 1


class Genome:
    def __init__(self):
        self.key = 1
        self.fitness = 0
        self.connection_genes = {}
        self.node_genes = {}
        # self.ancestors = {}

    def copy(self):
        g = Genome()
        g.key = self.key
        g.fitness = self.fitness
        g.connection_genes = copy.deepcopy(self.connection_genes)
        g.node_genes = copy.deepcopy(self.node_genes)
        return g

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
        curr_connection_count = 1
        for in_gene in input_genes:
            for j, out_gene in enumerate(output_genes):
                e = ConnectionGene()
                e.weight = random.uniform(0, 1)
                e.innovation_num = innovation_num
                e.in_node_key = in_gene.node_number
                e.out_node_key = out_gene.node_number
                genome.connection_genes[curr_connection_count] = e
                update_innovation_number()
                curr_connection_count += 1
        return genome


def calculate_num_excess_disjoint_genes(genome1: set, genome2: set):
    disjoint = genome1.difference(genome2).union(genome2.difference(genome1))

    disjoint_count = 0
    excess_count = 0
    for node_num in disjoint:
        disjoint_count = disjoint_count + 1 if node_num in genome1 and min(genome2) < node_num < max(
            genome2) else disjoint_count
        disjoint_count = disjoint_count + 1 if node_num in genome2 and min(genome1) < node_num < max(
            genome1) else disjoint_count
        excess_count = excess_count + 1 if node_num in genome1 and min(genome2) > node_num or node_num > max(
            genome2) else excess_count
        excess_count = excess_count + 1 if node_num in genome2 and min(genome1) > node_num or node_num > max(
            genome1) else excess_count

    return excess_count, disjoint_count


def calculate_weight_difference(genome1, genome2):
    genome1_set = set(genome1.connection_genes)
    genome2_set = set(genome2.connection_genes)
    shared_conns = genome1_set.intersection(genome2_set)

    weight_dif = 0
    for conn_num in shared_conns:
        weight_dif += genome2.connection_genes[conn_num].weight - genome1.connection_genes[conn_num].weight
    return weight_dif


def compatibility_distance(genome1: Genome, genome2: Genome):
    genome1_set = set(genome1.node_genes)
    genome2_set = set(genome2.node_genes)
    max_genes = max([len(genome1_set), len(genome2_set)])

    N = max_genes if max_genes > 20 else 1
    excess, disjoint = calculate_num_excess_disjoint_genes(genome1_set, genome2_set)

    weight_difference = calculate_weight_difference(genome1, genome2)

    constant_excess = int(
        config['DefaultGenome']['constant_excess'])  # coefficients to adjust importance of certain factors
    constant_disjoint = int(config['DefaultGenome']['constant_disjoint'])
    constant_weight_difference = int(config['DefaultGenome']['constant_weight_difference'])

    distance = constant_excess * excess / N + constant_disjoint * disjoint / N + constant_weight_difference * weight_difference

    return distance


# Best performing r% of each species is randomply mated to generate Nji offspring,
# replacing the entire population of the species
def fitness():
    Nj = None  # num of old individuals
    Nji = None  # num of individuals in species j
    fij = None  # adjusted fitness of individual i in species j
    f = 1  # mean adjusted fitness in the entire population
    Nji = sum(Nj * fij) / f
    pass


# matching genes inherited randomly
# disjoint genes (in middle) and excess genes (ends of genome) are inherited from more fit parent
def crossover(genome1: Genome, genome2: Genome):
    offspring = Genome()

    set1 = set(genome1.connection_genes)
    set2 = set(genome2.connection_genes)

    total_set = set1.union(set2)
    disjoint = total_set.difference(set1).union(total_set.difference(set2))
    inner = set1.intersection(set2)

    for connect in total_set:
        # randomly inherit matching genes
        if connect in inner:
            chosen_genome = genome1 if random.getrandbits(1) is 1 else genome2
            offspring.connection_genes[connect] = chosen_genome.connection_genes[connect]
            offspring.node_genes[connect] = chosen_genome.node_genes[connect]
        # inherit genes from more fit
        if connect in disjoint:
            chosen_genome = genome1 if genome1.fitness > genome2.fitness else genome2
            offspring.connection_genes[connect] = chosen_genome.connection_genes[connect]
            offspring.node_genes[connect] = chosen_genome.node_genes[connect]

    return offspring


# add a new connection with a random weight to two previously unconnected nodes
# connection to new node is 1, from new node to forward node is same as current weight
def mutate_add_connection(genome: Genome):
    global innovation_num
    adjacency_matrix = np.zeros((len(genome.node_genes) ** 2,))

    # Go through all connections and fill in edge matrix
    for index, conn in genome.connection_genes.items():
        adjacency_matrix[conn.in_node_key * conn.out_node_key] = 1

    available = [i for i, val in enumerate(adjacency_matrix) if int(val) is 0]

    chosen_loc = available[random.randint(0, len(available) - 1)] + 1

    connect_node_from = int(chosen_loc / len(genome.node_genes))
    connect_node_to = chosen_loc % len(genome.node_genes)

    new_connection = ConnectionGene()
    new_connection.innovation_num = innovation_num
    new_connection.weight = 1
    new_connection.in_node_key = connect_node_from
    new_connection.out_node_key = connect_node_to
    new_connection.enabled = True

    genome.connection_genes[innovation_num] = new_connection
    update_innovation_number()


# split existing connection and place new node in between
def mutate_add_node(genome: Genome):
    global innovation_num
    # pick a random connection and add a new node
    connection_keys = list(genome.connection_genes.keys())
    index_loc = random.randint(0, len(connection_keys) - 1)
    conn_num_to_split = connection_keys[index_loc]

    # disable old connection
    genome.connection_genes[conn_num_to_split].enabled = False

    # Create 2 new connections
    new_connection_new_to_old = ConnectionGene()
    new_connection_old_to_new = ConnectionGene()
    new_gene = NodeGene(nodenum=len(genome.node_genes) + 1, nodetype=NodeType.Hidden)
    genome.node_genes[len(genome.node_genes) + 1] = new_gene

    # new connection from previous node to new node
    new_connection_old_to_new.innovation_num = innovation_num
    new_connection_old_to_new.out_node_key = new_gene.node_number
    new_connection_old_to_new.in_node_key = genome.connection_genes[conn_num_to_split].in_node_key
    new_connection_old_to_new.weight = 1
    update_innovation_number()

    # new connection from new gene to downstream node
    new_connection_new_to_old.innovation_num = innovation_num
    new_connection_new_to_old.in_node_key = new_gene.node_number
    new_connection_new_to_old.out_node_key = genome.connection_genes[conn_num_to_split].out_node_key
    new_connection_new_to_old.weight = genome.connection_genes[conn_num_to_split].weight
    update_innovation_number()

    # add new connections to genome
    genome.connection_genes[innovation_num + 1] = new_connection_old_to_new
    genome.connection_genes[innovation_num + 2] = new_connection_new_to_old

    return genome


def sort_species(genomes: []):
    dict = OrderedDict()

    curr_dict_key = 0
    while len(genomes) > 0:
        compare_key = random.randrange(0, len(genomes)-1) if len(genomes) > 2 else 0
        # compare_key = compare_key+1 if compare_key

        # didnt find any close species, create new
        if dict.get(curr_dict_key, None) is None:
            dict[curr_dict_key] = [genomes.pop(compare_key)]
            curr_dict_key = 0
        elif same_species(genomes[compare_key], dict[curr_dict_key][0]) is True:
            # print(f"same species, {compatibility_distance(genomes[0], dict[curr_dict_key][0])}")
            dict[curr_dict_key].append(genomes.pop(compare_key))
            curr_dict_key = 0
        else:
            curr_dict_key += 1
    return dict


def same_species(genome1, genome2):
    dist = compatibility_distance(genome1, genome2)
    if dist > 0:
        same = dist - float(config['DefaultGenome']['compatibility_threshold']) <= 0
    else:
        same = dist + float(config['DefaultGenome']['compatibility_threshold']) >= 0
    return same


