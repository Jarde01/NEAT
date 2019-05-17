import copy
import pytest

from main import GenomeFactory, compatibility_distance, mutate_add_node, config, mutate_add_connection, sort_species, \
    calculate_num_excess_disjoint_genes, NodeGene, NodeType, ConnectionGene, update_innovation_number, crossover, \
    create_population

innovation_num = 0

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
        mutate_add_node(g1)
        mutate_add_connection(g1)
        mutate_add_connection(g1)
        mutate_add_connection(g1)
        mutate_add_connection(g1)
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

    config['DefaultGenome']['compatibility_threshold'] = '100'

    genomes = []
    for i in range(num_genomes):
        g = GenomeFactory.create_genome(num_input, num_output)
        mutate_add_connection(g)
        genomes.append(g)

        g1 = g.copy()
        mutate_add_node(g1)
        genomes.append(g1)

        g2 = g.copy()
        genomes.append(g2)
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
    new_connection_index = num_input * num_output + 1

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
    new_connection_key = list(genome.connection_genes.keys())[new_connection_index-1]
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


def test_create_population():
    length = '10'
    config['NEAT']['pop_size'] = length

    g = GenomeFactory.create_genome(1,1)

    pop = create_population(g)

    assert(len(pop) == int(length))
