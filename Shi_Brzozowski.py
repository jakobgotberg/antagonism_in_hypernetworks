import time,argparse,sys,itertools,copy, random
import numpy as np
import matrix_utils as mu
import hypergraph_algebra as hga

def generate_hypergraphs(n, pr, m=None, edges=[2,3]):
    '''
    Genereate Shi Brzozowski type hypergraphs
    'n' is the order of H: |V|
    'pr' is the edge probability.
    '''

    assert n >= 3, "n too small"
    assert pr >= 0.0, "probability must be non-negative"
    Is = []

    k = max(edges)
    m = n if m is None else m
    edges = [0] + edges
    weights = [1-pr if e == 0 else pr/len(edges) for e in edges]
    _1 = mu.one(n)
    rng = np.random.default_rng(time.thread_time_ns())

    inc = 0.05
    roof = 1 + inc
    for antagonism in np.arange(0.0, roof, inc):
        while True:
            c = np.array(random.choices(edges, weights=weights, k=m),dtype=int) * np.sign((np.random.rand(m) - antagonism)).astype(np.int8)
            I = np.zeros((n,m),dtype=int)
            for ix,k in enumerate(c):
                col = np.zeros(n)
                for e in np.array(np.random.choice(range(n),size=abs(k), replace=False)):
                    col[e] = random.choices([-1,1], k=1, weights=[antagonism, 1-antagonism])[0]
                I[:,ix] = col
        
            # Remove all zero columns
            cols_removed = sum(np.all(I == 0, axis=0))
            I = I[:, ~np.all(I == 0, axis=0)]

            # I have to check that it's connected.
            es = I.shape[1]
            #L = I @ I.T
            block_adj = np.concatenate([np.block([np.zeros((n,n)),I]), np.block([I.T, np.zeros((es,es))])])
            block_L = np.diag( (block_adj @ mu.one(block_adj.shape[0])).ravel() ) - block_adj
            fielder_eigenvalue = sorted(np.linalg.eigvals(block_L))[1]
            if not np.isclose(fielder_eigenvalue, 0.0, atol=1e-9) and np.unique(abs(I), axis=1).shape[1] == es-cols_removed:
                # Connected and not a multigraph
                del block_adj, block_L
                break
            
        Is.append(I)

    #assert np.isin(np.unique(Is), [-1,0,1]).all(), f"Is contains: {np.unique(Is)}"
    return Is

    
def main():

    p = argparse.ArgumentParser(exit_on_error=True)
    p.add_argument("--nodes", type=int, default=12)
    p.add_argument("--prob", type=float, default=0.6)
    p.add_argument("--normal",action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--abs",action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--edges", type=int, nargs="+", default=[2,3])
    a = p.parse_args()
    Incidences = generate_hypergraphs(a.nodes, a.prob, a.edges)
    for I in Incidences:
        print(I)
        print(f"Balanced: {hga.balanced_incidence(I)}")


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    t1 = time.perf_counter() - t0
    print(f"\n --- Total runtime: {t1:.3f}")
