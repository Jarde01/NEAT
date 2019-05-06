import tensorflow as tf
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


class GeneType(enum):
    Sensor = 1
    Output = 2
    Hidden = 3

class Gene:
    def __init__(self):
        self.In = ''
        self.Out = ''
        self.Weight = 0.0
        self.Enabled = True
        self.Innov = None  # help to find corresponding genes


class Genome:
    def __init__(self):
        self.key = 1
        self.fitness = None
        self.connection_genes = {}
        self.node_genes = {}
        # self.ancestors = {}

    def add(self, node1, node2):
        pass

    def mutate(self):
        pass


def create(self, num_genomes, type=None, config=None):


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
