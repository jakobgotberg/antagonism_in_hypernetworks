import random, collections, math
import numpy as np
import matrix_utils as mu
import utils

def unique_hyperedges(n, cards_of_edges, antagonism, rng):
    '''
    Generates an oriented hypergraph that is not a multigraph.
    Not necessarily connected. 
    '''

    assert isinstance(cards_of_edges, np.ndarray)
    assert isinstance(n, int) and n > 0
    assert 0 <= antagonism <= 1.0

    def unrank(n, cardinality, rank):
        '''
        Each rank represent a combinations in comb(n, cardinality),
        each rank is a unique hyperedge of a given cardinality.

        k := cardinality

        (v_1, v_2, v_3, ..., v_{k-1}, v_k) -> rank_0
        (v_2, v_3, v_4, ..., v_k, v_{k+1}) -> rank_1
        ...
        (v_k, v_{k+1}, v_{k+2}, ..., v_{n-1}, v_n) -> rank_m

        m := utils.comb(n, cardinality) -1

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
                combination_id = utils.comb(n - v - 1, vertices_remaining - 1)
                if rank < combination_id:
                    vertices_in_edge.append(v)
                    vertex = v + 1
                    break
                rank = rank - combination_id

        return np.array(vertices_in_edge, dtype=int)
    
    m = len(cards_of_edges)
    assert sum(cards_of_edges) >= n, f"Cannot cover all vertices if {sum(cards_of_edges)} < {n}"
    M = np.zeros((m,n), dtype=int)

    for card, count in collections.Counter(cards_of_edges).items():

        # The support of a vector is the number of combinations we can get 
        # with its non-zero elements.
        total_supports = utils.comb(n, card)
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

    assert {type(j) for i in I[:] for j in i[:]} == {np.int64}, f"{n}, {cards_of_edges}, {antagonism}, {rng} - { {type(j) for i in I[:] for j in i[:]} }"
    assert np.unique(abs(I), axis=1).shape[1] == I.shape[1], "Multigraph"
    return I

def generate_hypergraphs(n, cardinalities=[2,3], increment=0.05, verbose=False):
    '''
    Genereate Shi Brzozowski type hypergraphs.
    'n' is |V|.
    'cardinalities' are the possible hyperedge cardinalities.
    'increment' is the increment of antagonism per iteration.


     While improbable, it is possible that the generation never finds a connected hypergraph,
        hence, the program should be used with a timer, see utils.py, where the generation is 
        canceled if it takes too long, i.e., if it is impossible to generate a graph with the
        given arguments.

    The function is fine-tuned for n (:V|) in [12,22].

    '''

    def sigmoid(n,):
        min_c = 20
        max_c = min_c + 10
        max_  = max_c - min_c
        min_n, max_n = 12, 22
        middle = max_n - min_n

        return math.floor( ( max_ / ( 1 + math.e**(-0.5* (n - 2*middle)) ) ) + min_c )

    LARGEST_ANTAGONISM = 1.0

    assert isinstance(n, int) and n >= 3
    assert isinstance(increment, float) and 0.0 < increment < LARGEST_ANTAGONISM

    assert isinstance(cardinalities, list) and len(cardinalities) > 0 and \
        all(isinstance(c, int) for c in cardinalities) and \
        len(cardinalities) == len(set(cardinalities)) and \
        all(2 <= c <= n for c in cardinalities), f"{cardinalities}, {n}"

    if len(cardinalities) == 1 and cardinalities[0] == n:
        print("Legal but boring graph")
        assert False


    largest_number_of_possible_edges = sum(utils.comb(n,c) for c in cardinalities)
    # The largest possible |E| is too large for NP-Complete measures unless |V| is very small.
    # The sigmoid function scales down the max to be in [20,30], scaling with |V|.
    largest_number_of_possible_edges = sigmoid(n) if largest_number_of_possible_edges > sigmoid(n) \
            else largest_number_of_possible_edges

    Is = []
    roof = LARGEST_ANTAGONISM + increment
    rng = np.random.default_rng()

    for antagonism in np.arange(0.0, roof, increment):
        while True:

            # More probably to select higher |E|
            t = random.randint(3, largest_number_of_possible_edges)
            w = [2**i for i in range(3, t+2)]
            m = random.choices(list(range(3, t+2)), weights=w, k=1)[0]

            # Random selection of edges' cardinalities, smaller cardinalities are prefered to create
            # more interesting graphs.
            n_c = len(cardinalities)
            w=[1.414**i for i in range(n_c, 0, -1)]
            cards_of_edges = np.array(random.choices(cardinalities, weights=w, k=m), dtype=int)

            if sum(c-1 for c in cards_of_edges) < n-1:
                # Can't create connected hypergraph.
                continue

            if not np.all([count <= utils.comb(n, card) for card, count in collections.Counter(cards_of_edges).items()]):
                # Can't create edges of cardinality and quantity.
                continue

            I = unique_hyperedges(n, cards_of_edges, antagonism, rng)

            A = np.abs(I) @ np.abs(I.T)
            np.fill_diagonal(A,0)
            if not mu.irreducible(A):
                # Not connected.
                continue

            if verbose:
                print(f"Cards: {cards_of_edges}")

            break
            
        Is.append(I)

    for I in Is:
        assert np.isin(np.unique(I), [-1,0,1]).all(), f"The matrices contains illegal entires: {np.unique(Is)}"
    return Is

