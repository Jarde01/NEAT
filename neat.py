import random
from collections import OrderedDict

import NeuralNetwork
from config import Config
from genome import Genome, GenomeFactory
from enums.node_type import NodeType

xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]


class Population:
    def __init__(self):
        self.species = OrderedDict()


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
        Config.config['DefaultGenome']['constant_excess'])  # coefficients to adjust importance of certain factors
    constant_disjoint = int(Config.config['DefaultGenome']['constant_disjoint'])
    constant_weight_difference = int(Config.config['DefaultGenome']['constant_weight_difference'])

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


import copy


# matching genes inherited randomly
# disjoint genes (in middle) and excess genes (ends of genome) are inherited from more fit parent
def crossover(genome1: Genome, genome2: Genome):
    offspring = Genome()
    offspring.node_genes = copy.deepcopy(genome1.node_genes)
    offspring.node_genes.update(genome2.node_genes)

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
        # inherit genes from more fit
        if connect in disjoint:
            chosen_genome = genome1 if genome1.fitness > genome2.fitness else genome2
            offspring.connection_genes[connect] = chosen_genome.connection_genes[connect]

    return offspring


# split existing connection and place new node in between


def sort_species(genomes: []):
    dict = OrderedDict()

    curr_dict_key = 0
    while len(genomes) > 0:
        compare_key = random.randrange(0, len(genomes) - 1) if len(genomes) > 2 else 0
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
        same = dist - float(Config.config['DefaultGenome']['compatibility_threshold']) <= 0
    else:
        same = dist + float(Config.config['DefaultGenome']['compatibility_threshold']) >= 0
    return same


def create_population(init_genome: Genome):
    pop = []
    for x in range(0, int(Config.config['NEAT']['pop_size'])):
        g = init_genome.copy()
        g.mutate_modify_weights() if random.uniform(0, 1) < float(
            Config.config['DefaultGenome']['bias_mutate_rate']) else g
        g.mutate_add_connection() if random.uniform(0, 1) < float(
            Config.config['DefaultGenome']['bias_mutate_rate']) else g
        g.mutate_add_node() if random.uniform(0, 1) < float(Config.config['DefaultGenome']['bias_mutate_rate']) else g
        pop.append(g)
    return pop


def count_node_types(genome: Genome):
    num_in, num_hid, num_out = 0, 0, 0
    for key, node_gene in genome.node_genes.items():
        if node_gene.node_type == NodeType.Hidden:
            num_hid += 1
        elif node_gene.node_type == NodeType.Input:
            num_in += 1
        elif node_gene.node_type == NodeType.Output:
            num_out += 1
    return num_in, num_hid, num_out


# To implement:
# reporters: prints out information about stuff
# fitness functions
# Running the sample

'''
1. Initial population: take single genome then duplicate with mutations
2. Create model networks
3. evaluation genomes
4. speciate
5. take top 2 from each species and reproduce into a single genome
6. take single genome then duplicate with mutations
'''

# x = [[0, 0], [0, 1], [1, 1], [1, 0]]
# y = [[0], [1], [0], [1]]
#
# g = GenomeFactory.create_genome(2, 1)
# pop = create_population(g)
#
# results = []
# for genome in pop:
#     results.append(sum(NeuralNetwork.NeuralNetwork.feedforward(genome=genome, x_input=x, y_out=y)))
#
# print()
