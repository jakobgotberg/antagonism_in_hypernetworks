import itertools, collections
import matrix_utils as mu
import numpy as np

'''
    Measurements on signed hypergraphs ans signed graphs.
'''


def wang_balanced(L, allowed_error = 0.01):
    '''
    Wang et al. (2021) Theorem 3.5.
    G is balanced if the maximum single value is an eigenvalue of G's Laplacian.
    G must be connected.
    '''
    assert L.shape[0] == L.shape[1], "L is not a square matrix"
    rho = sorted(np.linalg.svd( np.abs(L), compute_uv=False))[-1]
    eigs = sorted(np.linalg.eigvals(L))[-1]
    close = True if any(np.abs(rho - e) <= allowed_error for e in eigs) else False
    #print(f"close: {rho - eigs[-1]:.3f}" if close else f"not close: {rho - eigs[-1]}")
    return True if close else False

def maximum_balance(I):
    _, m = I.shape
    for k in range(m + 1):
        for deleted_set in itertools.combinations(range(m), k):
            if balanced_incidence(np.delete(I,deleted_set,axis=1)):
                return k
    return m

#def RHO(L):
#    '''
#    |rho(|L|) - rho(L)|
#    '''
#    assert L.shape[0] == L.shape[1], f"L is not a square matrix"
#    abs_rho = mu.rho(np.abs(L))
#    rho = mu.rho(L)
#    return abs(abs_rho - rho)
#
#def SVD(M):
#    '''
#    |max_svd(|M|) - max_svd(M)|
#    '''
#    max_abs_svd = mu.max_svd(np.abs(M))
#    max_svd = mu.max_svd(M)
#    return abs(max_abs_svd - max_svd)

def incidence_to_abs_pairwise_adjacency(I):
    '''
    Returns the pair-wise adjacency representation. 
    '''
    A = np.abs(I) @ np.abs(I).T
    np.fill_diagonal(A,0)
    return A

def rho_is_closest(L):
    '''
    Asserts if rho(|L|) is closest to rho(L) and that
    rho(|L|) is larger than or equal to rho(L), within floating point
    approximation error.
    '''
    assert L.shape[0] == L.shape[1], f"L is not a square matrix"
    eigs = sorted(np.linalg.eigvals(L))
    rho = eigs[-1]
    abs_rho = mu.max_svd(np.abs(L))
    assert rho == min(eigs, key=lambda z: abs(z - abs_rho)), f"{eigs} --- {rho}:{abs_rho}"
    assert (rho - abs_rho).real < 1e-9, f"{rho}, {abs_rho}"

def maxsvd_is_closest(M):
    '''
    Asserts if the max absolute single value (max_abs_svd) is closest to the 
    max single value (max_svd) of L. 
    Additionally, that max_abs_svd is larger than or equal to max_svd, 
    within floating point approximation error.
    '''
    svds = sorted(np.linalg.svd(M, compute_uv=False))
    max_svd = svds[-1]
    max_abs_svd = mu.max_svd(np.abs(M))
    assert max_svd == min(svds, key=lambda z: abs(z - max_abs_svd)), f"{svds} --- {max_svd}:{max_abs_svd}"
    assert (max_svd - max_abs_svd).real < 1e-9, f"{max_svd}, {max_abs_svd}"

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

def balanced_incidence(I):
    '''
    Balanced as defined by Shi and Brzozowski.
    The hypergraph can be unconnected.
    '''
    assert isinstance(I, np.ndarray)
    # n = |V|, m = |E|
    n, m = I.shape

    # One list of adjacent nodes for each node
    signed_adj = collections.defaultdict(list)
    # need to distinguish pairwise connection to prevent going back on the same
    # exact pairwise connection in any hyperedge in the graph traversal.
    pairwise_index = 0

    for e in range(m):
        # get all vertices in the hyperedge, i.e., all non-zero elements
        # in the e:th column.
        incident = np.nonzero(I[:, e])[0]

        # Go through all pairs in hyperedge and compute the edge sign
        for pairwise in itertools.combinations(incident, r=2):
            u,v = pairwise
            sign = int(I[u, e] * I[v, e]) # cast from np.int64 to int.
            signed_adj[u].append((v, sign, pairwise_index))
            signed_adj[v].append((u, sign, pairwise_index))
            pairwise_index += 1

    def impossible_to_bipartition(start):
        partitioning = {}
        '''
        Uses DFS to go through hypergraph and partitions each node into group 0 or group 1.
        (The start node's group is arbitrary for 2-partitioning.)
        Start node is assigned group 0, all negatively adjacent neighbors are assigned group 1.

        If the algorithm finds an already visited node, there is a cycle,
        if the node is in one group, but the 'next_group' is the other group,
        the cycle is negative and the bipartition is impossible, i.e., unbalanced.
        '''

        # (node, group, origin relation)
        stack = [(start, 0, -1)]
        while stack:
            node, group, origin_connection = stack.pop()
            if node in partitioning:
                if partitioning[node] != group:
                    # Inconsistency, cannot bipartition.
                    return True
                continue
            partitioning[node] = group
            for neighbour, sign, pairwise_connection in signed_adj[node]:
                if pairwise_connection == origin_connection:
                    # Skip parent connection.
                    continue
                next_group = group if sign == 1 else 1 - group
                stack.append((neighbour, next_group, pairwise_connection))
        return False

    for i in range(n):
        if impossible_to_bipartition(i):
            return False
    return True

