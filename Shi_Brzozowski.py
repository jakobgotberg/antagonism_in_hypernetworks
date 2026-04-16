import random, math, time
import numpy as np
import matrix_utils as mu

def generate_hypergraphs(n, m=None, cardinalities=[2,3], increment=0.05):
    '''
    Genereate Shi Brzozowski type hypergraphs.
    'n' is |V|.
    'm' is |E|.
    'cardinalities' are the possible hyperedge cardinalities.
    'increment' is the increment of antagonism per iteration.

    Run program with a timeout if using m << n, 
        especially if forall c in cardinalities, c << n and m << n.
    '''
    LARGEST_ANTAGONISM = 0.5

    assert isinstance(n, int) and n >= 3
    assert isinstance(increment, float) and 0.0 < increment < LARGEST_ANTAGONISM

    assert isinstance(cardinalities, list) and cardinalities is not [] and \
        all(isinstance(c, int) for c in cardinalities) and \
        len(cardinalities) == len(set(cardinalities)) and \
        all(2 <= c <= n for c in cardinalities)

    m = n if m is None else m
    largest_number_of_possible_edges = sum(math.comb(n,c) for c in cardinalities)
    assert isinstance(m, int) and 0 < m <= largest_number_of_possible_edges, f"{m} > {largest_number_of_possible_edges}. |V| = {n}, cards = {cardinalities}"
    # The size of m = |E| must be larger than some value related to n = |V| for the graph to be
    #   connected, but this is not straightforward to compute when len(cardinalities) != 1.

    # The program should be used with a timer, see utils.py, where the generation is 
    #   canceled if it takes too long, i.e., if it is impossible to generate a graph with the
    #   given arguments.

    Is = []
    roof = LARGEST_ANTAGONISM + increment
    for antagonism in np.arange(0.0, roof, increment):

        while True:
            # Random selection of edges' cardinalities, each cardinality have the same probability/weight.
            t0 = time.perf_counter()
            cards = np.array(random.choices(cardinalities, k=m), dtype=int)
            
            # Generate random lists in {-1,0,1}^n, where the number of non-zeros = the cardinality.
            edges = []
            for c in cards:
                edge = np.zeros(n, dtype=int)
                random_vertex_gen = np.random.choice(range(n), size=c, replace=False)
                for e in np.array(random_vertex_gen):
                    edge[e] = random.choices([-1,1], k=1, weights=[antagonism, 1-antagonism])[0]
                edges.append(edge)

            I = np.array(edges).T

            if not np.unique(abs(I), axis=1).shape[1] == I.shape[1]:
                #no edges shall be identical.
                continue

            A = I @ I.T
            np.fill_diagonal(A,0)
            if mu.irreducible(A):
                break
            
        Is.append(I)

    assert np.isin(np.unique(Is), [-1,0,1]).all(), f"The matrices contains illegal entires: {np.unique(Is)}"
    return Is

