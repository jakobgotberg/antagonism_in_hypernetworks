import numpy as np
import matrix_utils as mu
import networkx as nx

def generate_graphs(n, pr, cyclic=False):
    '''
    'cyclic' means that no graph shall be a tree; trees are always balanced.
    'n' is the order
    'pr' is the edge probability.
    '''
    assert n >= 2, "n too small."
    assert pr > 0.0, "probability must be positive."

    As = []
    for antagonism in np.arange(0.0, 1+0.1, 0.05):
        attempt = 1
        while True:
            A = np.triu((np.random.rand(n,n) < pr).astype(np.int64), k=1) * \
                    np.sign(np.triu(np.random.rand(n,n) - antagonism)).astype(np.int64)
            A = A + A.T

            # irreducible and not False <=> irreducible
            # or, if graph must be cyclic, irreducible and not a tree
            if mu.irreducible(np.abs(A)) and not (nx.is_tree(nx.from_numpy_array(np.abs(A))) if cyclic else False):
                assert (A == A.T).all()
                break
            # Try again if graph is not irreducible, i.e., not connected.

        As.append(A)

    assert np.isin(np.unique(As), [-1,0,1]).all(), f"Contains: {np.unique(As)}"
    return As
