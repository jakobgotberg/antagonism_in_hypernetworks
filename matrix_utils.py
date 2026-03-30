import math, signal, itertools, random, copy, sys
import numpy as np
import numpy.linalg as linalg
cp = copy.deepcopy
    

def one(n):
    assert isinstance(n , int)
    assert n > 0
    return np.array([[1]] * n)

def square(B):
    return True if width(B) == hight(B) else False

def n_(B):
    assert square(B), "Not square"
    return hight(B)

def rho(B):
    assert isinstance(B, np.ndarray)
    return sorted(np.linalg.eigvals(B))[-1]

def max_svd(B):
    assert isinstance(B, np.ndarray)
    return sorted(np.linalg.svdvals(B))[-1]

def nonnegative(B):
    assert isinstance(B, np.ndarray)
    return not any([b < 0 for b in B.ravel()])

def positive(B):
    assert isinstance(B, np.ndarray)
    return not any([b <= 0 for b in B.ravel()])

def row_stochastic(B):
    if not nonnegative(B):
        return False
    One = one(width(B))
    return True if all(B @ One) == all(One) else False

def semi_convergent(B):
    pass
    # return rho(B) == 1 ## is this true?

def convergent(B):
    return rho(B) < 1

def irreducible(B):
    return positive( _matrix_power_sum(B, 0, n_(B)-1) )

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

def absolute_bipartite_incidence_adjacency(I):
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
    return sorted(np.linalg.eigvals(L))[1]

def matrix_power_sum(B, power):
    return _matrix_power_sum(B, 1, power)

def _matrix_power_sum(B, start, end):
    assert start < end, "Start >= End"
    assert isinstance(B, np.ndarray)
    assert square(B), "Matrix not square"

    n = hight(B)
    S = np.zeros((n,n))
    for k in range(start, end):
        S += linalg.matrix_power(B, k)
    return S

