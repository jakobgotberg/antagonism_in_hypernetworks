import time
import random, itertools
from collections import defaultdict
import matrix_utils as mu
import numpy as np


'''
    Helper
'''

def get_H(A):
    '''
    We define the degree matrix as: A2 + 1_3.T @ A3_1 + ... + 1_3.T @ A3_n
    No pairwise connection in this program, hence A2 is the zero matrix
    H = inv(D) @ adjacency
    '''
    n = A.shape[0]
    _1 = mu.one(n)
    A_tilde       =  np.zeros((n,n)) + np.array([(_1.T @ A3).ravel() for A3 in T[:]]) 
    #if not absolute else np.zeros((n,n)) + np.array([(_1.T @ abs(A3)).ravel() for A3 in T[:]])

    if not a_tilde:
        degree_matrix =  np.diag( (abs(A_tilde) @ _1).ravel() )

        # The normalized adjacency matrix, H
        return np.linalg.inv(degree_matrix) @ A_tilde

    return A_tilde

'''
    Measurements
'''
def algebraic_conflict(H):
    return sorted(np.linalg.eigvals(H))[-1]

def compute_fi(H):
    '''
    Find S through brute forcing all elements: s in {-1,1}^n
    The returned value is normalized: not an integer. How can I convert it to one?
    '''
    n = H.shape[0]
    _1 = mu.one(n)
    fi = lambda S : _1.T @ (abs(H) - (S @ H @ S)) @ _1
    return min([fi(np.diag(s)) for s in itertools.product([-1,1], repeat=n)]).ravel()

def maximum_balance(I, verbose=False):
    assert isinstance(I, np.ndarray)
    #assert np.issubdtype(I.dtype, np.integer)
    n, m = I.shape
    for k in range(m + 1):
        if verbose:
            print(k)
        for deleted_set in itertools.combinations(range(m), k):
            if balanced_incidence(np.delete(I,deleted_set,axis=1)):
                return k
    return m

def RHO(L):
    '''
    |rho(|L|) - rho(L)|
    '''
    assert L.shape[0] == L.shape[1], f"L is not a square matrix"
    abs_rho = sorted(np.linalg.eigvals( np.abs(L) ))[-1]
    rho = sorted(np.linalg.eigvals(L))[-1]
    return abs(abs_rho - rho)

def SVD(M):
    '''
    |max_svd(|M|) - max_svd(M)|
    '''
    max_abs_svd = sorted(np.linalg.svdvals( np.abs(M) ))[-1]
    max_svd = sorted(np.linalg.svdvals(M))[-1]
    return abs(max_abs_svd - max_svd)

def rho_is_closest(L):
    '''
    Asserts if rho(|L|) is closest to rho(L)
    '''
    assert L.shape[0] == L.shape[1], f"L is not a square matrix"
    abs_rho = sorted(np.linalg.eigvals( np.abs(L) ))[-1]
    eigs = sorted(np.linalg.eigvals(L))
    rho = eigs[-1]
    return rho == min(eigs, key=lambda z: abs(z - abs_rho))

def maxsvd_is_closest(L):
    '''
    Asserts if the max absolute single value is closest to the max single value
    of L.
    '''
    assert L.shape[0] == L.shape[1], f"L is not a square matrix"
    svds = sorted(np.linalg.svdvals(L))
    max_sv = svds[-1]
    max_abs_svd = sorted(np.linalg.svdvals( np.abs(L) ))[-1]
    return max_sv == min(svds, key=lambda z: abs(z - max_abs_svd))

def max_se(M):
    '''
    Graph must be connected.
    M is Wang et al style incidence.
    Returns relative number of negative edges in lowest 
        signature transformation.

    Returns a value between 0 and 1, the min ratio of 
        negative incidences over all incidences.

        Zero means balanced.
        A value close to 1 means very unbalanced.
        A value of 1 is not possible as the all negative graph can be
            transformed into the all possitve one. 
    '''
    e,v = M.shape
    minimum = np.finfo(np.float64).max
    Mp = np.abs(M)
    p_sum = np.sum(Mp)
    for sf, su in itertools.product( itertools.product([-1,1],repeat=e), itertools.product([-1,1],repeat=v) ):
        Sf = np.diag(sf)
        Su = np.diag(su)
        f = ((Sf @ M @ Su) == -1).sum() / p_sum
        if f == 0.0:
            return 0.0
        if f < minimum:
            minimum = f

    return minimum




def balanced_incidence(I, verbose=False):
    '''
    Balanced as defined by Shi and Brzozowski.
    The hypergraph can be unconnected.
    '''
    assert isinstance(I, np.ndarray)
    #assert np.issubdtype(I.dtype, np.integer)
    n, m = I.shape

    # One list of adjacent nodes for each node
    signed_adj = defaultdict(list)
    # need to distinguish pairwise connection to prevent going back on the same
    # exact pairwise connection in any hyperedge in the graph traversal.
    pairwise_index = 0

    for e in range(m):
        # get all vertices in the hyperedge
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

    time_spent = lambda i: print(f"{i if i else ''} {time.perf_counter() - t0:.3f} s") if verbose else lambda : None
    t0 = time.perf_counter()
    for i in range(n):
        if impossible_to_bipartition(i):
            time_spent("Unbalanced")
            return False
        if i % 100 == 0:
            time_spent(i)
    time_spent("Balanced")
    return True

