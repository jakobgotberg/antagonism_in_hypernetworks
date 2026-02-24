import time,argparse,sys,itertools,copy
import numpy as np
import matrix_utils as mu
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def assert_legal_A3(T):
    '''
    Iterate through all possible non-zero entreis in A3_i and assert that the same entries are covered in the relavent A3_j and A3_k, 
    With the same sign
    
    There is no way to escape O(n^3). Each element should only be checked once: which is optimal.
    As long as n is small, T can be in the L1 cache and the lookups are very fast desipte the strides.
    '''

    n = T.shape[0]
    # A generator with unique triples: no two triples are permutations of each other.
    g = ((i,j,k) for i,j,k in itertools.product(range(n), repeat=3) if i<j<k)

    # len({...}) == 1: the length of the set of values of entires in A3 is 1 if all entries are equal.
    lookup = lambda p : T[p[0]][p[1]][p[2]]
    assert_edge = lambda permutations : len({lookup(p) for p in permutations}) == 1  

    try:
        while hyperedge:=next(g):
            i,j,k = hyperedge
            # calls assert_edge with a generator object: no wasted memory, only wasted CPU cycles
            assert assert_edge( itertools.permutations(hyperedge) ), \
                                        f"A3 is not legal for {hyperedge}:\
                                          {T[i][j][k]},\
                                          {T[i][k][j]},\
                                          {T[j][i][k]},\
                                          {T[j][k][i]},\
                                          {T[k][i][j]},\
                                          {T[k][j][i]}"

    except StopIteration:
        return

def get_H(T, absolute=False):
    '''
    We define the degree matrix as: A2 + 1_3.T @ A3_1 + ... + 1_3.T @ A3_n
    No pairwise connection in this program, hence A2 is the zero matrix
    H = inv(D) @ adjacency
    '''
    n = T.shape[0]
    _1 = mu.one(n)
    A_tilde       =  np.zeros((n,n)) + np.array([(_1.T @ A3).ravel() for A3 in T[:]]) if not absolute \
                        else np.zeros((n,n)) + np.array([(_1.T @ abs(A3)).ravel() for A3 in T[:]])
    degree_matrix =  np.diag( (abs(A_tilde) @ _1).ravel() )

    # The normalized adjacency matrix, H
    return np.linalg.inv(degree_matrix) @ A_tilde


def compute_fi(H):
    '''
    Find S through brute forcing all elements: s in {-1,1}^n
    The returned value is normalized: not an integer. How can I convert it to one?
    '''
    n = H.shape[0]
    _1 = mu.one(n)
    fi = lambda S : (1/6) * _1.T @ (abs(H) - (S @ H @ S)) @ _1
    return min([fi(np.diag(s)) for s in itertools.product([-1,1], repeat=n)]).ravel()



def generate_hypergraphs(n, pr):
    '''
    Generate a list with an increasing amount of negative hyperedges.
    Erdős–Rényi hypergraphs?
    'n' is the order
    'pr' is the edge probability.
    '''

    assert n >= 3, "n too small"
    assert pr >= 0.0, "probability must be non-negative"
    r = range(n)
    pr = pr / n
    _1 = mu.one(n)
    set_val = lambda T,p,val : T[p[0]][p[1]][p[2]] = val

    def set_edges(T,A,i):
        r,c = np.nonzero(A)
        g = ((j,k,val) for zip(r,c,A[r,c]))

    # len({...}) == 1: the length of the set of values of entires in A3 is 1 if all entries are equal.
        for jk in itertools.product(r,r):
            j,k = jk
            if A[j][k] != 0:
                for v in itertools.permutations([i,j,k]):
                    T[v[0]][v[1]][v[2]] = A[j][k]
                    

    Ts = []
    for antagonism in np.arange(0.0, 1+0.2, 0.2):
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
            set_edges(T,node)

        assert_legal_A3(T)
        Ts.append(T)

    return Ts

    
def main():

    p = argparse.ArgumentParser(exit_on_error=True)
    p.add_argument("--verbose",action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--nodes", type=int, default=12)
    p.add_argument("--prob", type=float, default=0.6)
    p.add_argument("--normal",action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--abs",action=argparse.BooleanOptionalAction, default=False)
    a = p.parse_args()

    print("Since 'H' is the normalized adjacency, balance should mean \rho(H) = 1. Anything below 1 means unbalanced.")
    for T in generate_hypergraphs(a.nodes, a.prob):
        H = get_H(T, a.abs)
        algebraic_conflict = max(np.linalg.eigvals(H))
        print(f"lambda_n: {algebraic_conflict}")
        print(f"FI: {compute_fi(H)}\n")

if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    t1 = time.perf_counter() - t0
    print(f"\n --- Total runtime: {t1:.3f}")
