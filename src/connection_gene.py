import random


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