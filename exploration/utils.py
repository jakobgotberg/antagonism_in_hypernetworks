import random
import math
import numpy as np

    

def random_grs(n):
    '''
    geometric random row stochastic squere matrix:
    '''
    def row():
        r = [random.random()**i for i in range(n)]
        s = math.fsum(r)
        w = [x/s for x in r]
        assert math.isclose(math.fsum(w), 1.0, rel_tol=1e-12, abs_tol=1e-15)
        return w

    return np.array([row() for _ in range(n)])
    
def is_nonnegative(B):
    return not any([b < 0 for b in B.ravel()])


