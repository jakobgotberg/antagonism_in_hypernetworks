import signal, itertools, sys
import numpy as np
import utils

'''
    Matrices and some matrix algebra.
'''

'''
    In use
'''
def square(B):
    return True if B.shape[0] == B.shape[1] else False

def get_E(A):
    assert isinstance(A, np.ndarray)
    E = np.sum(np.triu(np.abs(A), k=1) == 1)
    return E


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

def negative_incidence_product_ratio(I):
    '''
    Ratio of different pair-wise incidence relations.

    Per edge:
        # different sign = #positives * #negatives
        # same sign = binom(#positives, 2) + binom(#negatives, 2)
    '''
    M = I.T
    same = 0
    diff = 0

    for edge in M:
        positives = np.sum(edge == 1)
        negatives = np.sum(edge == -1)
        diff += positives * negatives
        same += utils.comb(positives, 2) + utils.comb(negatives, 2)
        
    return diff / (same + diff)


def rho(B, assert_psd=False):
    assert isinstance(B, np.ndarray)
    if assert_psd:
        assert positive_semidefinite(B)
    return sorted(np.linalg.eigvals(B))[-1]

def max_svd(B):
    assert isinstance(B, np.ndarray)
    return sorted(np.linalg.svd(B, compute_uv=False))[-1]

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

def positive_semidefinite(B):
    assert square(B)
    assert all(e >= -1e-10 for e in np.linalg.eigvals(B))
    return True

def irreducible(B):
    return positive( _matrix_power_sum(B, 0, B.shape[0]-1) )

def _matrix_power_sum(B, start, end):
    assert start < end, "Start >= End"
    assert isinstance(B, np.ndarray)
    assert square(B), "Matrix not square"

    n = B.shape[0]
    S = np.zeros((n,n))
    for k in range(start, end+1):
        S += np.linalg.matrix_power(B, k)
    return S

def normalized_pairwise_adjacency(A):
    assert (np.diag(A) == 0).all(), "Not an adjacency matrix"
    assert (A == A.T).all(), "Not a symmetric graph"
    n = A.shape[0]
    try:
        degree_matrix =  np.diag(np.abs(A) @ np.ones(n))
        H = np.linalg.inv(degree_matrix) @ A
        assert square(H)
        assert row_stochastic(np.abs(H)), "Not abs row stochastic"
        return H
    except np.linalg.LinAlgError:
        print(f"Sys.exit(): Singular matrix?\n{np.diag(degree_matrix)}")
        sys.exit()

def row_stochastic(B):
    if not nonnegative(B):
        print("non")
        return False
    return np.allclose(B @ np.ones(B.shape[1]), np.ones(B.shape[0]))

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
            B = np.linalg.matrix_power(A,k)
            if (B > 0).all():
                return True
            k += 1

    except Timeout:
        return False
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

def normal_algebraic_conflict(H):
    '''
    Algebraic conflict on the (normalized) adjacency matrix.
    From Aref et al.
    '''
    assert (np.diag(H) == 0).all(), "Not an adjacency matrix"
    assert row_stochastic(np.abs(H)), "Not abs row stochastic"

    degrees = sorted((H != 0).astype(np.int64) @ np.ones(H.shape[0]))
    d_max = (degrees[-1] + degrees[-2]) / 2

    return 1 - (algebraic_conflict(H) / (d_max -1))

def algebraic_conflict(H):
    '''
    Algebraic conflict on the (normalized) adjacency matrix.
    '''
    assert (np.diag(H) == 0).all(), "Not an adjacency matrix"
    assert row_stochastic(np.abs(H)), "Not abs row stochastic"
    return 1 - rho(H)

def normal_FI(H):
    '''
    From Aref et al.
    '''
    assert (np.diag(H) == 0).all(), "Not an adjacency matrix"
    assert row_stochastic(np.abs(H)), "Not abs row stochastic"
    number_of_edges = np.sum(np.triu( (H != 0).astype(np.int64) ))
    return FI(H) / (number_of_edges)

def FI(H):
    '''
    Find S through brute forcing all elements: s in {-1,1}^n
    The returned value is normalized.
    '''
    assert (np.diag(H) == 0).all(), "Not an adjacency matrix"
    assert row_stochastic(np.abs(H)), "Not abs row stochastic"
    n = H.shape[0]
    fi = lambda S : np.ones(n) @ (np.abs(H) - (S @ H @ S)) @ np.ones(n)
    return min([fi(np.diag(s)) for s in itertools.product([-1,1], repeat=n)])


