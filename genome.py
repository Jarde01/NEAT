import copy
import random
from collections import defaultdict

import numpy as np

from enums.node_type import NodeType
from config import Config
from utils import InnovationNumber
from connection_gene import ConnectionGene
from node_gene import NodeGene

class Genome:
    def __init__(self):
        self.key = 1
        self.fitness = 0
        self.connection_genes = {}
        self.node_genes = {}
        self.inputs = []
        self.outputs = []
        self.hiddens = []

    def copy(self):
        g = Genome()
        g.key = self.key
        g.fitness = self.fitness
        g.connection_genes = copy.deepcopy(self.connection_genes)
        g.node_genes = copy.deepcopy(self.node_genes)
        g.inputs = copy.deepcopy(self.inputs)
        g.outputs = copy.deepcopy(self.outputs)
        g.hiddens = copy.deepcopy(self.hiddens)
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

    def create_graphs(self):
        network = defaultdict(list)
        rev_network = defaultdict(list)

        for innov, conn in self.connection_genes.items():
            network[conn.in_node_key].append(conn.out_node_key)
            rev_network[conn.out_node_key].append(conn.in_node_key)

        return network, rev_network

    def list_connections(self):
        conns = []
        for innov, conn in self.connection_genes.items():
            conns.append((conn.in_node_key, conn.out_node_key))
        return conns

    def find_connection(self, in_node, out_node):
        for conn in self.connection_genes.values():
            if conn.in_node_key == in_node and conn.out_node_key == out_node:
                return conn
        return None

    def crossover(self, parent_genome):
        offspring = Genome()
        offspring.node_genes = copy.deepcopy(self.node_genes)
        offspring.node_genes.update(parent_genome.node_genes)

        set1 = set(self.connection_genes)
        set2 = set(parent_genome.connection_genes)

        total_set = set1.union(set2)
        disjoint = total_set.difference(set1).union(total_set.difference(set2))
        inner = set1.intersection(set2)

        for connect in total_set:
            # randomly inherit matching genes
            if connect in inner:
                chosen_genome = self if random.getrandbits(1) is 1 else parent_genome
                offspring.connection_genes[connect] = chosen_genome.connection_genes[connect]
            # inherit genes from more fit
            if connect in disjoint:
                chosen_genome = self if self.fitness > parent_genome.fitness else parent_genome
                offspring.connection_genes[connect] = chosen_genome.connection_genes[connect]

        return offspring

    def mutate_add_node(self):
        # pick a random connection and add a new node
        connection_keys = list(self.connection_genes.keys())
        index_loc = random.randint(0, len(connection_keys) - 1)
        conn_num_to_split = connection_keys[index_loc]

        # disable old connection
        self.connection_genes[conn_num_to_split].enabled = False

        # Create 2 new connections
        new_connection_new_to_old = ConnectionGene()
        new_connection_old_to_new = ConnectionGene()
        new_gene = NodeGene(nodenum=len(self.node_genes) + 1, nodetype=NodeType.Hidden)
        self.hiddens.append(new_gene.node_number)
        self.node_genes[len(self.node_genes) + 1] = new_gene

        # new connection from previous node to new node
        new_connection_old_to_new.innovation_num = InnovationNumber.innovation_num
        new_connection_old_to_new.out_node_key = new_gene.node_number
        new_connection_old_to_new.in_node_key = self.connection_genes[conn_num_to_split].in_node_key
        new_connection_old_to_new.weight = 1
        InnovationNumber.innovation_num += 1

        # new connection from new gene to downstream node
        new_connection_new_to_old.innovation_num = InnovationNumber.innovation_num
        new_connection_new_to_old.in_node_key = new_gene.node_number
        new_connection_new_to_old.out_node_key = self.connection_genes[conn_num_to_split].out_node_key
        new_connection_new_to_old.weight = self.connection_genes[conn_num_to_split].weight
        InnovationNumber.innovation_num += 1

        # add new connections to genome
        self.connection_genes[InnovationNumber.innovation_num + 1] = new_connection_old_to_new
        self.connection_genes[InnovationNumber.innovation_num + 2] = new_connection_new_to_old
        return

    # TODO: currently we can create cycles when adding a connection, only connect from hidden to hidden/output
    # add a new connection with a random weight to two previously unconnected nodes
    # connection to new node is 1, from new node to forward node is same as current weight
    def mutate_add_connection(self):
        num_inputs = len(self.inputs)
        num_outputs = len(self.outputs)
        num_nodes = len(self.node_genes)
        adjacency_matrix = np.zeros(num_nodes*num_nodes).reshape(num_nodes, num_nodes)

        # Go through all connections and fill in edge matrix
        for index, conn in self.connection_genes.items():
            adjacency_matrix[conn.in_node_key][conn.out_node_key] = 1

        # available = [i for i, val in enumerate(adjacency_matrix) if int(val) is 0]
        # chose a location to attach a new connection to, minus the input nodes
        # chosen_loc = available[random.randint(len(self.inputs), len(available) - 1)] + 1
        # connect_node_from = int(chosen_loc / len(self.node_genes))
        # connect_node_to = chosen_loc % len(self.node_genes)

        possible_from_nodes = [x for x in self.hiddens+self.inputs]
        possible_to_nodes = [x for x in self.hiddens+self.outputs]

        possible_combos = []
        for from_node in possible_from_nodes:
            for to_node in possible_to_nodes:
                possible_combos.append((from_node, to_node))

        found = False
        while len(possible_combos)>0:
            connect_node_from, connect_node_to = possible_combos.pop(random.randint(0, len(possible_combos)-1))
            # Simple connecting to itself case
            if adjacency_matrix[connect_node_from][connect_node_to] != 1:
                found = True
                break

        if not found:
            return
        new_connection = ConnectionGene()
        new_connection.innovation_num = InnovationNumber.innovation_num
        new_connection.weight = 1
        new_connection.in_node_key = connect_node_from
        new_connection.out_node_key = connect_node_to
        new_connection.enabled = True

        self.connection_genes[InnovationNumber.innovation_num] = new_connection
        InnovationNumber.innovation_num += 1

    def mutate_modify_weights(self):
        for conn in self.connection_genes.values():
            conn.weight = random.uniform(-1, 1) * float(Config.config['DefaultGenome']['bias_mutate_power'])


class GenomeFactory:
    @staticmethod
    def create_genome(num_input_nodes, num_output_nodes):
        genome = Genome()

        curr_gene_num = 0
        input_genes = []
        for x in range(num_input_nodes):
            n = NodeGene(nodenum=curr_gene_num, nodetype=NodeType.Input)
            genome.node_genes[curr_gene_num] = n
            input_genes.append(n)
            genome.inputs.append(curr_gene_num)
            curr_gene_num += 1

        output_genes = []
        for x in range(num_output_nodes):
            n = NodeGene(nodenum=curr_gene_num, nodetype=NodeType.Output)
            output_genes.append(n)
            genome.node_genes[curr_gene_num] = n
            genome.outputs.append(curr_gene_num)
            curr_gene_num += 1

        # connect all the input to the output genes
        curr_connection_count = 0
        for in_gene in input_genes:
            for j, out_gene in enumerate(output_genes):
                e = ConnectionGene()
                e.weight = random.uniform(0, 1)
                e.innovation_num = InnovationNumber.innovation_num
                e.in_node_key = in_gene.node_number
                e.out_node_key = out_gene.node_number
                genome.connection_genes[curr_connection_count] = e
                InnovationNumber.innovation_num += 1
                curr_connection_count += 1
        return genome
