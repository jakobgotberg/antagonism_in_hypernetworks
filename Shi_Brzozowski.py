import random, math, collections
import numpy as np
import matrix_utils as mu
import utils


def unique_hyperedges(n, cards_of_edges, antagonism, rng):
    '''
    Generates an oriented hypergraph that is not a multigraph.
    Not necessarily connected. 
    '''

    assert isinstance(rng, np.random._generator.Generator)
    assert isinstance(cards_of_edges, np.ndarray)
    assert isinstance(n, int) and n > 0
    assert 0 <= antagonism <= 1.0

    def unrank(n, cardinality, rank):
        '''
        Each rank represent a combinations in math.comb(n, cardinality),
        each rank is a unique hyperedge of a given cardinality.

        k := cardinality

        (v_1, v_2, v_3, ..., v_{k-1}, v_k) -> rank_0
        (v_2, v_3, v_4, ..., v_k, v_{k+1}) -> rank_1
        ...
        (v_k, v_{k+1}, v_{k+2}, ..., v_{n-1}, v_n) -> rank_m

        m := math.comb(n, cardinality) -1

        unrank() takes the given rank and returns the hyperedge.

        e.g., n := 5, cardinality := 3, all possible hyperedges,

        [0,1,2] -> rank 0
        [0,1,3] -> rank 1
        [0,1,4] -> rank 2
        [0,2,3] -> rank 3
        [0,2,4] -> rank 4
        [0,3,4] -> rank 5
        [1,2,3] -> rank 6
        [1,2,4] -> rank 7
        [1,3,4] -> rank 8
        [2,3,4] -> rank 9

        '''
        vertices_in_edge = []
        vertex = 0
        for vertices_remaining in range(cardinality, 0, -1):

            for v in range(vertex, n):
                combination_id = math.comb(n - v - 1, vertices_remaining - 1)
                if rank < combination_id:
                    vertices_in_edge.append(v)
                    vertex = v + 1
                    break
                rank = rank - combination_id

        return np.array(vertices_in_edge)
    
    m = len(cards_of_edges)
    assert sum(cards_of_edges) >= n, f"Cannot cover all vertices if {sum(cards_of_edges)} < {n}"
    M = np.zeros((m,n), dtype=int)

    for card, count in collections.Counter(cards_of_edges).items():

        # The support of a vector is the number of combinations we can get 
        # with its non-zero elements.
        total_supports = math.comb(n, card)
        assert count <= total_supports, f"Impossible to generate {count} unique edges with card = {card} when |V| = {n}."

        # Rows where there shall be card vertices.
        edge_indexes = np.where(cards_of_edges == card)[0]

        # See unrank().
        ranks = rng.choice(total_supports, size=count, replace=False)

        for edge_id, rank in zip(edge_indexes, ranks):
            vertices_in_edge = unrank(n, card, rank)
            polarities = rng.choice([-1,1], size=card, p=[antagonism, 1-antagonism])
            M[edge_id][vertices_in_edge] = polarities

    I = M.T
    assert np.unique(abs(I), axis=1).shape[1] == I.shape[1], "Multigraph"
    return I

def generate_hypergraphs(n, cardinalities=[2,3], increment=0.05):
    '''
    Genereate Shi Brzozowski type hypergraphs.
    'n' is |V|.
    'cardinalities' are the possible hyperedge cardinalities.
    'increment' is the increment of antagonism per iteration.


     While improbable, it is possible that the generation never finds a connected hypergraph,
        hence, the program should be used with a timer, see utils.py, where the generation is 
        canceled if it takes too long, i.e., if it is impossible to generate a graph with the
        given arguments.

    '''

    LARGEST_ANTAGONISM = 0.5

    assert isinstance(n, int) and n >= 3
    assert isinstance(increment, float) and 0.0 < increment < LARGEST_ANTAGONISM

    assert isinstance(cardinalities, list) and cardinalities is not [] and \
        all(isinstance(c, int) for c in cardinalities) and \
        len(cardinalities) == len(set(cardinalities)) and \
        all(2 <= c <= n for c in cardinalities)

    largest_number_of_possible_edges = sum(math.comb(n,c) for c in cardinalities)
    #largest_card = cardinalities[-1]
    #lowest_connected_hypergraph = math.ceil( (n-1) / (largest_card-1) ) 

    Is = []
    roof = LARGEST_ANTAGONISM + increment
    rng = np.random.default_rng()

    for antagonism in np.arange(0.0, roof, increment):
        while True:
            # Random selection of edges' cardinalities, each cardinality have the same probability/weight.
            m = random.randint(1, largest_number_of_possible_edges)
            cards_of_edges = np.array(random.choices(cardinalities, k=m), dtype=int)

            if sum(c-1 for c in cards_of_edges) < n-1:
                # Can't create connected hypergraph.
                continue

            if not np.all([count <= math.comb(n, card) for card, count in collections.Counter(cards_of_edges).items()]):
                # Can't create edges of cardinality and quantity.
                continue

            I = unique_hyperedges(n, cards_of_edges, antagonism, rng)

            A = np.abs(I) @ np.abs(I.T)
            np.fill_diagonal(A,0)
            if not mu.irreducible(A):
                # Not connected.
                continue

            break
            
        Is.append(I)

    for I in Is:
        assert np.isin(np.unique(I), [-1,0,1]).all(), f"The matrices contains illegal entires: {np.unique(Is)}"
    return Is

