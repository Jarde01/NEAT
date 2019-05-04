innovation_number = 0

class Gene:
    def __init__(self):
        self.In = ''
        self.Out = ''
        self.Weight = 0.0
        self.Enabled = True
        self.Innov = '' # help to find corresponding genes


class Genome:
    def __init__(self):
        self.connection_genes = []

    def add(self, node1, node2):
        pass

    def mutate(self):
        pass


def speciation(genome1, genome2):
    N = max([len(genome1), len(genome2)])
    E = 1 # number of excess genes
    D = 1 # number of disjoint genes
    W = 0 # weight difference of matching genes
    c1 = 1 # coefficients to adjust importance of certain factors
    c2 = 1
    c3 = 1

    distance = c1*E/N + c2*D/N + c3*W


# Best performing r% of each species is randomply mated to generate Nji offspring,
# replacing the entire population of the species
def fitness():
    Nj = None # num of old individuals
    Nji = None # num of individuals in species j
    fij = None # adjusted fitness of individual i in species j
    f = None # mean adjusted fitness in the entire population
    Nji = sum(Nj*fij)/f
    pass