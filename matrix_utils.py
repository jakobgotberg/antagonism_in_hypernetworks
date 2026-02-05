import math, signal, itertools, random, copy
import numpy as np
import numpy.linalg as linalg
cp = copy.deepcopy
    
class Timeout(Exception):
    pass
def timeout_handler(signum, frame):
    raise Timeout

def e_i(n, i):
    return np.eye(n)[:, i]

def one(n):
    assert isinstance(n , int)
    assert n > 0
    return np.array([[1]] * n)

def PI_n(n):
    return np.eye(n) - ( (one(n)/n) @ one(n).T )

def square(B):
    return True if width(B) == hight(B) else False

def n_(B):
    assert square(B), "Not square"
    return hight(B)

def width(B):
    assert isinstance(B, np.ndarray)
    _, m = B.shape
    return m
def hight(B):
    assert isinstance(B, np.ndarray)
    n, _ = B.shape
    return n
def rho(B):
    assert isinstance(B, np.ndarray)
    return sorted(np.linalg.eigvals(B))[-1]

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


def averaging(A, x0, k):
    '''
    Returns xk = A^k x0
    '''
    return linalg.matrix_power(A, k) @ x0

def get_path_graph(n, self_loops=True):
    r = range(n)
    A = np.zeros((n,n))
    for ij in itertools.product(r,r):
        i,j = ij
        A[i][j] = 1 if i == j or i == j-1 or j == i-1 else 0
    return A if self_loops else A - np.diag(np.diag(A)) 

def get_random_uni(n):
    '''
    Returns a connected graph of size n
    Connected = irreducible for undirected graphs
    '''
    connected = False
    r = range(n)
    E = list(itertools.product(r,r))
    E = list(itertools.filterfalse(lambda e: e[0] >= e[1], E))
    F = cp(E)
    while not connected:
        E = list(itertools.filterfalse(lambda a:  random.randint(0,1), F))
        A = adjacency_matrix(E)
        connected = irreducible(A)
    
    return A

def wdir(n):
    A = np.array([
                [0, 1, 0, 0, 0],
                [1, 0, 0, 0, 1],
                [0, 1, 0, 0, 1],
                [0, 0, 1, 0, 0],
                [1, 0, 0, 1, 0]])
    return A

def get_random(n):
    import random
    '''
    Random row stochastic square matrix:
    '''
    def row():
        r = [random.randint(0,7) for i in range(n)]
        s = math.fsum(r)
        w = [x/s for x in r]
        assert math.isclose(math.fsum(w), 1.0, rel_tol=1e-12, abs_tol=1e-15)
        return w

    return np.array([row() for _ in range(n)])

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

def get_n_from_E(E:list):
    '''
    0 indexed
    '''
    return max([e[1] for e in E]) + 1
    
def check_E(E):
    assert isinstance(E, list)
    assert all(isinstance(t, tuple) for t in E)
    assert len(E) == len(set(E))
    # assert 0 indexed
    # assert no self-loops
    # assert a < b in (a,b) in E
    
def laplacian_matrix(E:list):
    return degree_matrix(E) - adjacency_matrix(E) 
def degree_matrix(E:list):
    '''
    E is the list of undirected edges, represented as tuples
    G is assumed to be connected
    '''
    check_E(E)
    n = get_n_from_E(E)
    D = [0] * n
    for e in E:
        D[e[0]] += 1
        D[e[1]] += 1
    return np.diag(D)
    
def wdir_adjacency_matrix(E:list):
    n = max(max([e[0] for e in E]), max([e[1] for e in E])) + 1
    A = np.zeros((n,n))
    for e in E:
        i, j, w = e
        A[i][j] = w
    return A

def adjacency_matrix(E:list):
    '''
    E is the list of undirected edges, represented as tuples
    G is assumed to be connected
    '''
    check_E(E)
    n = get_n_from_E(E)
    A = np.zeros((n,n))
    for e in E:
        i, j = e
        A[i][j] = A[j][i] = 1
    return A
