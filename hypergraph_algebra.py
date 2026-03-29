from __future__ import annotations
import time
from typing import Optional
import random, itertools
from collections import defaultdict
import matrix_utils as mu
import numpy as np

from itertools import combinations,product
from dataclasses import dataclass

from typing import Optional

abs_e = lambda e : tuple(abs(x) for x in e)



def get_signed_hyperedges(T, absolute=False):
    lookup = lambda T,p : T[p[0]][p[1]][p[2]]
    n = T.shape[0]
    E = []
    g = ((i,j,k) for i,j,k in itertools.product(range(n), repeat=3) if i<j<k)
    try:
        while hyperedge:=next(g):
            if sign:=lookup(T,hyperedge):
                i,j,k = hyperedge
                t = (int(sign*i),int(sign*j),int(sign*k)) if not absolute else (i,j,k)
                E.append(t)
    except StopIteration:
        pass

    return E



def get_incidence_tensor(Ts):
    Is = []
    for T in Ts:
        Is.append(get_incidence_matrix(T))
    return Is

def get_incidence_matrix(T):
    lookup = lambda T,p : T[p[0]][p[1]][p[2]]
    # extract all edges and put them into a
    n = T.shape[0]
    E = []
    g = ((i,j,k) for i,j,k in itertools.product(range(n), repeat=3) if i<j<k)
    try:
        while hyperedge:=next(g):
            if sign:=lookup(T,hyperedge):
                i,j,k = hyperedge
                col = np.zeros(n, dtype=int)
                col[i] = col[j] = col[k] = sign
                E.append(col)
    except StopIteration:
        pass

    return np.array(E).T



def wang_degree_se(L):
    '''
    Wang et al. (2021) Theorem 3.5.
    G must be connected.
    '''
    assert L.shape[0] == L.shape[1], "L is not a square matrix"
    rho = sorted(np.linalg.svdvals( np.abs(L) ))[-1]
    max_e = sorted(np.linalg.eigvals(L))[-1]
    return np.abs(rho - max_e)


def rho_maxrho_diff(L):
    '''
    |rho(|L|) - rho(L)|
    '''
    assert L.shape[0] == L.shape[1], f"L is not a square matrix"
    abs_rho = sorted(np.linalg.eigvals( np.abs(L) ))[-1]
    rho = sorted(np.linalg.eigvals(L))[-1]
    return abs(abs_rho - rho)

def maxsvd_maxabssvd_diff(L):
    '''
    |max_svd(|L|) - max_svd(L)|
    '''
    assert L.shape[0] == L.shape[1], f"L is not a square matrix"
    max_abs_svd = sorted(np.linalg.svdvals( np.abs(L) ))[-1]
    max_svd = sorted(np.linalg.svdvals(L))[-1]
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


def maxsvd_closest_to_maxeigenvalue(L):
    '''
    This is not what Wand et al. defined.
    '''
    assert L.shape[0] == L.shape[1], f"L is not a square matrix"
    rho = sorted(np.linalg.svdvals( np.abs(L) ))[-1]
    eigs = sorted(np.linalg.eigvals(L))
    max_e = eigs[-1]
    return max_e == min(eigs, key=lambda z: abs(z - rho))

def max_se(M, return_signature=False):
    '''
    Graph must be connected.
    M is Wang et al style incidence.
    Returns relative number of negative edges in lowest 
        signature transformation.

    '''
    e,v = M.shape
    minimum = np.finfo(np.float64).max
    min_sf = min_su = None
    Mp = np.abs(M)
    p_sum = np.sum(Mp)
    for sf, su in product( product([-1,1],repeat=e), product([-1,1],repeat=v) ):
        Sf = np.diag(sf); Su = np.diag(su)
        f = ((Sf @ M @ Su) == -1).sum() / p_sum
        if f < minimum:
            minimum = f
            if return_signature:
                min_sf = Sf; min_su = Su

    #return f if not return_signature else f, min_sf, min_su
    return minimum



def get_H(T, a_tilde=False):
    '''
    We define the degree matrix as: A2 + 1_3.T @ A3_1 + ... + 1_3.T @ A3_n
    No pairwise connection in this program, hence A2 is the zero matrix
    H = inv(D) @ adjacency
    '''
    n = T.shape[0]
    _1 = mu.one(n)
    A_tilde       =  np.zeros((n,n)) + np.array([(_1.T @ A3).ravel() for A3 in T[:]]) 
    #if not absolute else np.zeros((n,n)) + np.array([(_1.T @ abs(A3)).ravel() for A3 in T[:]])

    if not a_tilde:
        degree_matrix =  np.diag( (abs(A_tilde) @ _1).ravel() )

        # The normalized adjacency matrix, H
        return np.linalg.inv(degree_matrix) @ A_tilde

    return A_tilde


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

def hyper_frustration_index(T):

    E = get_signed_hyperedges(T)
    m = len(E)
    for k in range(m + 1):
        for keep in itertools.combinations(range(m), m-k):
            filtered = [e for ix,e in enumerate(E) if ix in keep]
            if _negative_hyperedge_cycles(filtered) == []:
                return k
    return m


def negative_hyperedge_cycles(T):
    E = get_signed_hyperedges(T)
    return _negative_hyperedge_cycles(E)

def _negative_hyperedge_cycles(E):
    '''
    Assumes list of hyperedges from a connected hypergraph.
    '''
    class Hyperedge:
        sign     = 0
        index    = -1
        vertices = None
        adjacent = defaultdict(list)

        def __init__(self, index, vertices):
            self.index = index
            self.vertices = vertices

        def add_adj(self, adjacent):
            self.adjacent = adjacent

        def __repr__(self):
            return f"{self.vertices}"

    class Negative_Cycle(Exception):
        pass
        path = None
        def __init__(self, path):
            self.path = path

    #def canonical_cycle(path):
    #    n = len(path)
    #    rots1 = [tuple(path[i:] + path[:i]) for i in range(n)]

    #    rev = path[::-1]
    #    rots2 = [tuple(rev[i:] + rev[:i]) for i in range(n)]

    #    return min(rots1 + rots2)

    m = len(E)
    visited = [None] * m
    HE = []
    for ix, e in enumerate(E):
        he = Hyperedge(ix, abs_e(e))
        he.sign = -1 if any(x < 0 for x in e) else 1
        HE.append(he)

    
    for h in HE:
        adjs = []
        for v in h.vertices:
            adjs.append([adj_h.index for adj_h in HE if adj_h.index != h.index and v in adj_h.vertices])

        # flatten list of lists
        adjs = [index for adjacent in adjs for index in adjacent]
        h.adjacent = set(adjs)

    def dfs(start,current,visited,path,parent):

        def negative_product(path):
            sign = 1
            for v in path:
                sign = sign * HE[v].sign
            return True if sign == -1 else False

        for adj in HE[current].adjacent:
            if adj == parent:
                continue
            if adj == start and len(path) >= 3 and negative_product(path):
                raise Negative_Cycle(path)
                    
            elif adj not in visited and adj >= start:
                visited.add(adj)
                path.append(adj)
                dfs(start,adj,visited,path,current)
                path.pop()
                visited.remove(adj)

    try:
        for start in [h.index for h in HE]:
            visited = {start}
            dfs(start,start,visited,[start], None)
    except Negative_Cycle as nc:
        return nc.path

    return []
