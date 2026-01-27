import math
import numpy as np
import numpy.linalg as linalg

    
def one(n):
    assert isinstance(n , int)
    assert n > 0
    return np.array([1] * n)

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
