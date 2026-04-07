import math, signal, itertools, random, copy, sys
import numpy as np
import numpy.linalg as linalg
cp = copy.deepcopy

'''
    Matrices and some matrix algebra.
'''


'''
    In use
'''
def square(B):
    return True if B.shape[0] == B.shape[1] else False

def negative_edge_ratio(A):
    '''
    Ratio of negative edges and all edges.
    Value in [0,1]. 0 -> no negative edges, 1 -> only negative edges.
    '''
    assert (np.diag(A) == 0).all(), "Not an adjacency matrix"
    assert np.isin(np.unique(A), [-1,0,1]).all() or \
    np.isin(np.unique(A), [0,1]).all() or np.isin(np.unique(A), [-1,0]).all()

    return np.sum(A == -1) / np.sum(np.abs(A))

def negative_incidence_ratio(I):
    '''
    Ratio of negative incidences and all incidences.
    Value in [0,1]. 0 -> no negative edges, 1 -> only negative edges.
    '''
    assert np.isin(np.unique(I), [-1,0,1]).all() or \
    np.isin(np.unique(I), [0,1]).all() or np.isin(np.unique(I), [-1,0]).all() \
    or np.isin(np.unique(I), [1]).all() or np.isin(np.unique(I), [-1]).all()

    return np.sum(I == -1) / np.sum(np.abs(I))

def rho(B):
    assert isinstance(B, np.ndarray)
    return sorted(np.linalg.eigvals(B))[-1]

def max_svd(B):
    assert isinstance(B, np.ndarray)
    return sorted(np.linalg.svdvals(B))[-1]

def positive(B):
    assert isinstance(B, np.ndarray)
    return (B > 0).all()

def nonnegative(B):
    assert isinstance(B, np.ndarray)
    return (B >= 0).all()

def Laplacian(A):
    '''
    Opposing Laplacian.
    '''
    assert (np.diag(A) == 0).all(), "Not an adjacency matrix"
    assert square(A), "Not an adjacency matrix"

    return np.diag(np.abs(A) @ np.ones(A.shape[0])) - A

def absolute_bipartite_incidence_adjacency(I):
    '''
    I in Z_{n,m}

    Ab = |0_{n,n},   I|
         |I.T, 0_{m,m}|
    
    Used to measure connectivity.
    '''

    I = np.abs(I)
    V = I.shape[0]
    E = I.T.shape[0]
    B1 = np.block([np.zeros((V,V),dtype=np.int8),I])
    B2 = np.block([I.T,np.zeros((E,E),dtype=np.int8)])
    B = np.concatenate([B1,B2])
    return B.astype(np.int8)

def absolute_bipartite_incidence_laplacian(I):
    A = absolute_bipartite_incidence_adjacency(I)
    D = np.diag(A @ np.ones(A.shape[0],dtype=np.int8)).astype(np.int8)
    L = D - A
    return L.astype(np.int8)

def fiedler(L):
    assert square(L), "Not a Laplacian"
    return sorted(np.linalg.eigvals(L))[1]

def irreducible(B):
    return positive( _matrix_power_sum(B, 0, B.shape[0]-1) )

def _matrix_power_sum(B, start, end):
    assert start < end, "Start >= End"
    assert isinstance(B, np.ndarray)
    assert square(B), "Matrix not square"

    n = B.shape[0]
    S = np.zeros((n,n))
    for k in range(start, end+1):
        S += linalg.matrix_power(B, k)
    return S

def normalized_pairwise_adjacency(A):
    assert (np.diag(A) == 0).all(), "Not an adjacency matrix"
    assert (A == A.T).all(), "Not a symmetric graph"
    n = A.shape[0]
    try:
        degree_matrix =  np.diag(np.abs(A) @ np.ones(n))
        H = np.linalg.inv(degree_matrix) @ A
        assert row_stochastic(np.abs(H))
        return H
    except np.linalg.LinAlgError:
        print(f"Singular matrix?\n{np.diag(degree_matrix)}")
        sys.exit()

def row_stochastic(B):
    if not nonnegative(B):
        return False
    m = B.shape[1]
    return True if np.all(B @ np.ones(m) == np.ones(m)) else False

'''
    Not in use
'''

def matrix_power_sum(B, power):
    return _matrix_power_sum(B, 1, power)

def convergent(B):
    return rho(B) < 1

class Timeout(Exception):
    pass
def timeout_handler(signum, frame):
    raise Timeout
def primitive(A, limit=10):
    '''
    An algorithm only checking 'primitivety' using the definition
    exists k in N s.t. A^k > 0
    might never halt, hence the 'limit'
    '''
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, limit)
    try:
        k = 1
        while True:
            B = linalg.matrix_power(A,k)
            if (B > 0).all():
                return True
            k += 1

    except Timeout:
        return False
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

