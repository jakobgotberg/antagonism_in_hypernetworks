import time,argparse,sys,itertools,copy
import numpy as np
import matrix_utils as mu
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def compute_balance(T, n, normalized=False):
    def find_S_with_A(T,s):
        '''
        The hypergraph is balanced if:
        Exists S : SA_iS <= 0 forall A_i

        However, it seems only the 100% cooperative hypergraph has this property
        '''
        S = np.diag(s)
        for ix, A in enumerate(T[:]):
            if ((S @ A @ S) < 0).any():
                return False
            # All matricies are non-negative, the hypergraph is balanced
            if ix == n-1:
                return True

    #def find_S_with_H(T,D_inv,s):
    #    pass
    #    '''
    #    No clue how the math works.
    #    Should H be H from "Collective Descision making [...]" ? In that case, there should be no loop
    #    '''

    #    S = np.diag(s)
    #    f_sum = 0
    #    for A in T[:]:
    #        H = D_inv @ A
    #        f_sum += _1.T @ (abs(H) - (S @ H @ S)) @ _1
    #    return f_sum


    # find S through brute forcing all elements: s in {-1,1}^n
    f_sum_min = sys.float_info.max
    s_best = None
    for s in itertools.product([-1,1], repeat=n):

        _T = copy.deepcopy(T)
        #f_sum = find_S_with_H(_T,D_inv,s) if normalized else find_S_with_A(_T,s)

        if find_S_with_A(_T,s):
            return True
        del _T
    #print(f"{antagonism:.1f} -> lowest frustration: {f_sum_min/2}")
    return False

def generate_hypergraphs(n, pr):
    '''
    Measure balance/frustration/albegraic conflict in 2-order hypergraphs with 'n' nodes.
    'pr' is the edge probability.
    '''

    assert n >= 3, "n too small"
    assert pr >= 0.0, "probability must be non-negative"
    r = range(n)
    #pr = pr / n
    _1 = mu.one(n)

    def set_edges(T,A,i):
        '''
        Set all permutations of the index (perm(3) = 6) to the value
        Can it be done faster? 100% yes.
        '''
        for jk in itertools.product(r,r):
            j,k = jk
            if A[j][k] != 0:
                for v in itertools.permutations([i,j,k]):
                    T[v[0]][v[1]][v[2]] = A[j][k]
                    

    Ts = []
    for antagonism in np.arange(0.0,0.1, 0.02):
        T = np.zeros((n,n,n))
        for node in range(n):
            A = T[node]
            # Iterate through all nodes, [0,n-1], assign each a signed random upper triangle.
            B = np.triu((np.random.rand(n,n) < pr).astype(np.int8), k=1) * np.sign(np.triu(np.random.rand(n,n) - antagonism)).astype(np.int8)
            # This ensures that entries made in previous iterations are not overwritten.
            mask = (A == 0) & (B != 0)
            A[mask] = B[mask]
            # Zero out the i:th column and row, 
            A[node,:] = 0
            A[:,node] = 0
            # Symmetry
            #A = A + A.T
            # No need to transpose, the 'set_edges' makes the matrix symmetric.
            set_edges(T,A,node)
        Ts.append(T)

        # How do we define degree for hypergraphs?
        #D_inv = np.eye(n)
        # Number of negative edges.
    return Ts

    
#def main():
#
#    p = argparse.ArgumentParser(exit_on_error=True)
#    p.add_argument("--verbose",action=argparse.BooleanOptionalAction, default=False)
#    p.add_argument("--nodes", type=int, default=12)
#    p.add_argument("--prob", type=float, default=0.6)
#    p.add_argument("--normal",action=argparse.BooleanOptionalAction, default=False)
#    a = p.parse_args()
#    measure(a.nodes, a.prob, a.normal)
#
#if __name__ == "__main__":
#    t0 = time.perf_counter()
#    main()
#    t1 = time.perf_counter() - t0
#    print(f"\n --- Total runtime: {t1:.3f}")
