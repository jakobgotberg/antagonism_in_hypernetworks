import random
from collections import defaultdict
import matrix_utils as mu
import numpy as np

def get_H(T, absolute=False):
    '''
    We define the degree matrix as: A2 + 1_3.T @ A3_1 + ... + 1_3.T @ A3_n
    No pairwise connection in this program, hence A2 is the zero matrix
    H = inv(D) @ adjacency
    '''
    n = T.shape[0]
    _1 = mu.one(n)
    A_tilde       =  np.zeros((n,n)) + np.array([(_1.T @ A3).ravel() for A3 in T[:]]) if not absolute \
                        else np.zeros((n,n)) + np.array([(_1.T @ abs(A3)).ravel() for A3 in T[:]])
    degree_matrix =  np.diag( (abs(A_tilde) @ _1).ravel() )

    # The normalized adjacency matrix, H
    return np.linalg.inv(degree_matrix) @ A_tilde, A_tilde


def compute_fi(H):
    '''
    Find S through brute forcing all elements: s in {-1,1}^n
    The returned value is normalized: not an integer. How can I convert it to one?
    '''
    n = H.shape[0]
    _1 = mu.one(n)
    fi = lambda S : _1.T @ (abs(H) - (S @ H @ S)) @ _1
    return min([fi(np.diag(s)) for s in itertools.product([-1,1], repeat=n)]).ravel()


def balanced_Incidence(I):
    '''
    Assumes a connected hypergraph.
    '''
    assert isinstance(I, np.ndarray)
    assert np.issubdtype(I.dtype, np.integer)
    n, m = I.shape

    # One list of adjacent nodes for each node
    signed_adj = defaultdict(list)
    # need to distinguish pairwise connection to prevent going back on the same
    # exact pairwise connection in any hyperedge in the graph traversal.
    pairwise_index = 0

    for e in range(m):
        incident = np.nonzero(I[:, e])[0]
        # Go through all pairs in hyperedge and compute the edge sign
        for i in range(len(incident)):
            for j in range(i + 1, len(incident)):
                u, v = int(incident[i]), int(incident[j])
                sign = int(I[u, e]) * int(I[v, e])
                signed_adj[u].append((v, sign, pairwise_index))
                signed_adj[v].append((u, sign, pairwise_index))
                pairwise_index += 1

    def impossible_to_bipartition(start):
        partitioning = {}
        '''
        Uses DFS to go through hypergraph and partitions each node into group 0 or group 1.
        The start node is arbitrary for 2-partitioning.
        start node is assigned 0; all negatively adjacent neighbors are assigned 1.

        If the algorithm finds an already visited node, there is a cycle:
        if the node is in one group, but the 'next_group' is the other group,
        the cycle is negative and the bipartition is impossible.
        '''

        # (node, group, origin relation)
        stack = [(start, 0, -1)]
        while stack:
            node, group, origin_connection = stack.pop()
            if node in partitioning:
                if partitioning[node] != group:
                    return True
                continue
            partitioning[node] = group
            for neighbour, sign, pairwise_connection in signed_adj[node]:
                if pairwise_connection == origin_connection:
                    continue
                next_group = group if sign == 1 else 1 - group
                stack.append((neighbour, next_group, pairwise_connection))
        return False

    return not impossible_to_bipartition(random.choice(range(n)))
