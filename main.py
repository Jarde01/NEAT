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
    # update_innovation_number()

    # new connection from new gene to downstream node
    new_connection_new_to_old.innovation_num = innovation_num
    new_connection_new_to_old.in_node_key = new_gene.node_number
    new_connection_new_to_old.out_node_key = genome.connection_genes[conn_num_to_split].out_node_key
    new_connection_new_to_old.weight = genome.connection_genes[conn_num_to_split].weight
    # update_innovation_number()

    # add new connections to genome
    genome.connection_genes[innovation_num + 1] = new_connection_old_to_new
    genome.connection_genes[innovation_num + 2] = new_connection_new_to_old

    return genome


def sort_species(genomes: []):
    dict = OrderedDict()

    dict[0] = [genomes.pop(0)]

    curr_dict_key = 0
    while len(genomes) is not 0:
        # didnt find any close species, create new
        if dict.get(curr_dict_key, None) is None:
            dict[curr_dict_key] = [genomes.pop(0)]
            curr_dict_key = 0
        elif same_species(genomes[0], dict[curr_dict_key][0]) is True:
            # print(f"same species, {compatibility_distance(genomes[0], dict[curr_dict_key][0])}")
            dict[curr_dict_key].append(genomes.pop(0))
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


def test_compatibility_distance():
    num_input = 1
    num_output = 1

    g = GenomeFactory.create_genome(num_input, num_output)
    g1 = g.copy()

    distance = compatibility_distance(g, g1)
    assert (distance < 1)
    assert (distance > -1)

    mutate_add_node(g1)
    mutate_add_node(g1)
    mutate_add_node(g1)

    distance = compatibility_distance(g, g1)
    assert (distance > 1 or distance > -1)


def test_sort_species_multiple():
    num_input = 3
    num_output = 2
    num_genomes = 5

    config['DefaultGenome']['compatibility_threshold'] = '0.1'

    genomes = []
    for i in range(num_genomes):
        g = GenomeFactory.create_genome(num_input, num_output)
        mutate_add_connection(g)
        mutate_add_node(g)
        mutate_add_node(g)
        mutate_add_node(g)
        genomes.append(g)

        g1 = g.copy()
        mutate_add_node(g)
        mutate_add_connection(g)
        mutate_add_connection(g)
        mutate_add_connection(g)
        mutate_add_connection(g)
        genomes.append(g1)

        g2 = g.copy()
        genomes.append(g2)
        # random.seed(random.randint(0,255))

    result = sort_species(genomes)
    assert (len(result.keys()) > 1)
    assert (len(result.keys()) <= num_genomes * 3)


def test_sort_species_single():
    num_input = 3
    num_output = 2
    num_genomes = 5

    genomes = []
    for i in range(num_genomes):
        g = GenomeFactory.create_genome(num_input, num_output)
        mutate_add_connection(g)
        genomes.append(g)

        g = GenomeFactory.create_genome(num_input, num_output)
        mutate_add_node(g)
        genomes.append(g)

        g = GenomeFactory.create_genome(num_input, num_output)
        genomes.append(g)
        # random.seed(random.randint(0,255))

    result = sort_species(genomes)

    assert (len(result.keys()) == 1)


def test_calculate_num_excess_disjoint_genes():
    # Excess
    genome1 = {1, 2, 3, 4, 5, 6, 7}
    genome2 = {2, 3, 4}
    excess, disjoint = calculate_num_excess_disjoint_genes(genome1, genome2)
    assert (excess == 4)
    assert (disjoint == 0)

    # One oe each end
    genome1 = {2, 3, 4, 5}
    genome2 = {1, 2, 3, 4}
    excess, disjoint = calculate_num_excess_disjoint_genes(genome1, genome2)
    assert (excess == 2)
    assert (disjoint == 0)

    # disjoint
    genome1 = {1, 5}
    genome2 = {1, 2, 3, 4, 5}
    excess, disjoint = calculate_num_excess_disjoint_genes(genome1, genome2)
    assert (excess == 0)
    assert (disjoint == 3)

    # disjoint and excess
    genome1 = {1, 5, 6, 7, 8}
    genome2 = {1, 2, 3, 4, 5}
    excess, disjoint = calculate_num_excess_disjoint_genes(genome1, genome2)
    assert (excess == 3)
    assert (disjoint == 3)


def test_mutate_add_connection():
    num_input = 3
    num_output = 2

    before = GenomeFactory.create_genome(num_input, num_output)
    genome = copy.deepcopy(before)

    mutate_add_connection(genome)

    assert (len(before.connection_genes) < len(genome.connection_genes))


def test_mutate_add_node():
    num_input = 3
    num_output = 2

    new_node_key = num_input + num_output + 1
    new_connection_key = num_input * num_output + 1

    before = GenomeFactory.create_genome(num_input, num_output)
    genome = copy.deepcopy(before)

    mutate_add_node(genome)

    # find and make sure one node is disabled
    loc = None
    for index, gene in genome.connection_genes.items():
        if not gene.enabled:
            loc = index
            break

    # Test gene is turned off
    assert (before.connection_genes[loc].enabled is True)
    assert (genome.connection_genes[loc].enabled is False)

    # Test new connections and nodes are present
    assert (len(genome.connection_genes) > len(before.connection_genes))
    assert (len(genome.connection_genes) == len(before.connection_genes) + 2)

    assert (len(genome.node_genes) > len(before.node_genes))
    assert (len(genome.node_genes) == len(before.node_genes) + 1)

    # make sure new connections are added with correct values
    assert (genome.connection_genes[new_connection_key])
    assert (genome.connection_genes[new_connection_key].enabled is True)

    # Make sure old connection and new connections are hooked up appropriately
    first_new_conn_in_node_key = genome.connection_genes[new_connection_key].in_node_key
    first_new_conn_out_node_key = genome.connection_genes[new_connection_key].out_node_key

    second_new_in_node_key = genome.connection_genes[new_connection_key + 1].in_node_key
    second_new_out_node_key = genome.connection_genes[new_connection_key + 1].out_node_key

    prev_in = before.connection_genes[loc].in_node_key
    prev_out = before.connection_genes[loc].out_node_key

    assert (first_new_conn_in_node_key == prev_in)
    assert (first_new_conn_out_node_key == new_node_key)

    assert (second_new_in_node_key == new_node_key)
    assert (second_new_out_node_key == prev_out)


def create_with_hidden_layer(genome2):
    global innovation_num
    new_node_key = 4
    new_out_node_key = 3
    conn_modifying_key = 2

    genome2.node_genes[new_node_key] = NodeGene(new_node_key, NodeType.Hidden)
    newcon = ConnectionGene()
    newcon.innovation_num = innovation_num
    update_innovation_number()
    newcon.in_node_key = new_node_key
    newcon.out_node_key = new_out_node_key
    genome2.connection_genes[new_out_node_key] = newcon
    genome2.connection_genes[conn_modifying_key].out_node_key = new_node_key

    return genome2


def create_disjoint_genomes():
    genome1 = GenomeFactory().create_genome(2, 1)
    genome1.fitness = 1
    genome2 = GenomeFactory().create_genome(2, 1)
    genome2.fitness = 3
    genome2_hid = create_with_hidden_layer(genome2)
    return genome1, genome2_hid


def test_crossover():
    genome1, genome2 = create_disjoint_genomes()
    offspring = crossover(genome1, genome2)
    assert (len(offspring.connection_genes.keys()) == 3)
    assert (offspring.connection_genes[3] == genome2.connection_genes[3])


def test_create_genome():
    tests = [(1, 1), (2, 1), (9, 9)]
    for num_i, num_o in tests:
        g1 = GenomeFactory.create_genome(num_i, num_o)
        connection_count = len(g1.connection_genes)

        print(f'Expected: {num_i * num_o}, Found: {connection_count}')
        assert (connection_count == num_i * num_o)


def test():
    out = ConnectionGene()
    out.out_node_key = None

    c1 = ConnectionGene()
    c1.in_node_key = 1
    c1.out_node_key = 3

    c2 = ConnectionGene()
    c2.in_node_key = 2
    c2.out_node_key = 3

    n1 = NodeGene(1, NodeType.Input)
    n2 = NodeGene(2, NodeType.Input)
    n3 = NodeGene(3, NodeType.Output)

    gene = Genome()
    gene.node_genes = {1: n1, 2: n2, 3: n3}
    gene.connection_genes = {1: c1, 2: c2, 3: out}


# test_speciation()
# test_compatibility_distance()
test_sort_species_single()
# test_sort_species_multiple()
