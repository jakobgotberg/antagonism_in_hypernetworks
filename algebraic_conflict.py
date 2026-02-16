import time,argparse
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


#% Working example (random interactions, with increasing number of negative edges)

# you can start with 10 nodes (to understand the code), but to do the figures and the analysis choose at least 20 nodes
# (otherwise, due to the random interactions, it is difficult to see the
# relationship between frustration and smallest eigenvalue of the signed
# Laplacian. NOTE that already with n= 20, the function "f_frustration"
# takes some time!
#num_neg_edges = 0:0.2:1; 
# Number of Negative Edges
def measure(n, pr):
    nne = np.arange(0, 1+0.2, 0.2) 
    len_nne = len(nne)
    def cell():
        return [None] * len_nne
    def get_adjacency(antagonism):
        while True:
            A = np.triu((np.random.rand(n,n) < pr).astype(np.int8), k=1) * np.sign(np.triu(np.random.rand(n,n) - antagonism))
            A = A + A.T
            cc, _ = connected_components(csr_matrix(A), directed=False)
            if cc == 1:
                return A

    adjacencies = [get_adjacency(antagonism) for antagonism in nne]

    for antagonism, A in zip(nne, adjacencies):
        # H is normalized sigend adjacency matrix
        # H = D^{-1} @ A
        # find H by solving D @ H = A
        D = np.diag(np.sum(np.abs(A), axis=1))
        H = np.linalg.solve(D, A)

        # Algebraic conflict
        alg = min(np.linalg.eigvals(np.eye(n) - H))
        # Number of Negative edges
        neg = sum(sum(A < 0))
        # Frustration, Energy of matrix, S?
        #f, e, s = f_frustration(H)
        print(f"{antagonism:.1f} -> {neg}; l1 = {alg:.3f}")
    
def main():

    p = argparse.ArgumentParser(exit_on_error=True)
    p.add_argument("--verbose",action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--nodes", type=int, default=12)
    p.add_argument("--prob", type=float, default=0.6)
    a = p.parse_args()
    measure(a.nodes, a.prob)

if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    t1 = time.perf_counter() - t0
    print(f"\n --- Total runtime: {t1:.3f}")
