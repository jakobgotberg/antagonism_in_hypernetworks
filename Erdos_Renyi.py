import time,argparse,sys,itertools,copy
import numpy as np
import matrix_utils as mu
import hypergraph_algebra as hga

def assert_legal_A3(T):
    '''
    Iterate through all possible non-zero entreis in A3_i and assert that the same entries are covered in the relavent A3_j and A3_k, 
    With the same sign
    
    There is no way to escape O(n^3). Each element should only be checked once: which is optimal.
    As long as n is small, T can be in the L1 cache and the lookups are very fast desipte the strides.
    '''
    f_str = lambda T,i,j,k : f"{T[i][j][k]}, {T[i][k][j]}, {T[j][i][k]}, {T[j][k][i]}, {T[k][i][j]}, {T[k][j][i]}\n"
    lookup = lambda T,p : T[p[0]][p[1]][p[2]]

    n = T.shape[0]
    # A generator with unique triples: no two triples are permutations of each other.
    g = ((i,j,k) for i,j,k in itertools.product(range(n), repeat=3) if i<j<k)

    # len({...}) == 1: the length of the set of values of entires in A3 is 1 if all entries are equal.
    assert_edge = lambda permutations : len({lookup(T,p) for p in permutations}) == 1  

    try:
        while hyperedge:=next(g):
            # calls assert_edge with a generator object: no wasted memory, only wasted CPU cycles
            assert assert_edge( itertools.permutations(hyperedge) ), \
                                        f"A3_{hyperedge[0]} is not legal for {hyperedge}: " + f_str(T, hyperedge[0], hyperedge[1], hyperedge[2])\
                                        + str(T[hyperedge[0]]) + "\n\n" \
                                        + str(T[hyperedge[1]]) + "\n\n" \
                                        + str(T[hyperedge[2]]) + "\n\n"

    except StopIteration:
        pass


def generate_hypergraphs(n, pr, restrict=False):
    '''
    Generate a list with an increasing amount of negative hyperedges.
    Erdős–Rényi hypergraphs?
    'n' is the order
    'pr' is the edge probability.
    '''


    assert n >= 3, "n too small"
    assert pr >= 0.0, "probability must be non-negative"
    Ts = []
    r = range(n)
    pr = pr / n
    _1 = mu.one(n)

    def set_edges(T,C,i):
        '''
        Go through all new entires in C (upper triangle) and, for each edge, set the same values in the two related 'A' matricies.
        Because of the mask, the function will not overwrite existing edges with a new signs, as far as I can tell.
        Each entry _should_ only be written to once.
        '''
        negative_edges = 0
        row, col = np.nonzero(C)
        g = (edge for edge in zip(row, col, C[row, col]))
        try:
            while edge:=next(g):
                j, k, val = edge
                negative_edges += 1 if val == -1 else 0
                T[i][j][k] = T[i][k][j] = T[j][i][k] = T[j][k][i] = T[k][i][j] = T[k][j][i] = val
        except StopIteration:
            return negative_edges

    for antagonism in np.arange(0.4, 1+0.1, 0.2):
        while True:
            T = np.zeros((n,n,n))
            for node in range(n):
                A = T[node]
                # Iterate through all nodes, [0,n-1], assign each a signed random upper triangle.
                B = np.triu((np.random.rand(n,n) < pr).astype(np.int8), k=1) * np.sign(np.triu(np.random.rand(n,n) - antagonism)).astype(np.int8)
                # Zero out the i:th column and row, 
                B[node,:] = 0
                B[:,node] = 0

                # This ensures that entries made in previous iterations are not overwritten.
                mask = (A == 0) & (B != 0)
                if (mask == False).all():
                    # No new edges added in this iteration.
                    continue

                # C holds the new, legal edges generated with B in its non-zero entries
                C = np.zeros((n,n))
                C[mask] = B[mask]
                set_edges(T, C, node)
                #A[mask] = B[mask]
                #A += A.T
                del B,C

            # No isolated nodes
            # This is for 2-order hyperedges exclusively


            if restrict:
                E = hga.get_signed_hyperedges(T, absolute=True)
                if any(sum(x in t for t in E) > 2 for x in {e for t in E for e in t}):
                    continue

            if mu.irreducible(abs(A_tilde)):
                break

        assert_legal_A3(T)
        Ts.append(T)

    assert np.isin(np.unique(Ts), [-1,0,1]).all(), f"Ts contains: {np.unique(Ts)}"
    return Ts

    
def main():

    p = argparse.ArgumentParser(exit_on_error=True)
    p.add_argument("--nodes", type=int, default=12)
    p.add_argument("--prob", type=float, default=0.6)
    p.add_argument("--abs",action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--show",action=argparse.BooleanOptionalAction, default=False)

    a = p.parse_args()

    for T in generate_hypergraphs(a.nodes, a.prob):

        E   = hga.get_signed_hyperedges(T)
        E_n = sum(1 for e in E if any(x < 0 for x in e))
        
        H, A_tilde = hga.get_H(T, a.abs)
        algebraic_conflict = max(np.linalg.eigvals(H))

        if a.show:
            I = hga.get_incidence_matrix(T)
            print(I)

        print(f"Edges: {len(E)}\t Negative Edges: {E_n}")
        print(f"lambda_n: {algebraic_conflict}")
        print(f"FI: {hga.compute_fi(H)}")
        if E_n:
            neg_cycl = hga.negative_hyperedge_cycles(T)
            if neg_cycl == True:
                hfi = hga.hyper_frustration_index(T)
                print(f"Hyper FI: {hfi}")
            else:
                assert neg_cycl is not None
                print("No negative cycles")

        print()

if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    t1 = time.perf_counter() - t0
    print(f"\n --- Total runtime: {t1:.3f}")
