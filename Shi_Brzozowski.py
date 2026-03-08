import time,argparse,sys,itertools,copy, random
import numpy as np
import matrix_utils as mu

def balanced(I):
    '''
    A decision problem: balanced or not balanced.
    Corollary 3.1: Checking the balance of H takes linear time

    The last checking takes O(|E|^2).
    '''

            

    # If col_i(a) == col_i(b), a and b are in the same partition, if we find a contradiction, col_j(a) != col_j(b),
    # I is not the incidence matrix of a balanced hypergraph.
    n = I.shape[0]
    m = I.shape[1]
    # 'p' for partition, True <-> in partition, False <-> not in partition, None <-> unassigned.
    p = {i: None for i in range(n)}

    # remaining hyperedges to explore are stored in 'he'
    # first element is the column index
    he = [(ix, *map(int, np.nonzero(col)[0])) for ix, col in enumerate(I.T)]
    next_branch = []

    # Put lowest index vertices in random edge in the partition
    e = random.choice(he)
    p[e[1]] = True

    st = lambda t,eq : p[t] if eq else not p[t]
    lookup = lambda pw : (p[pw[0]],p[pw[1]])

    while True:
        col_ix = e[0]
        pairs = list(itertools.combinations(e[1:], 2))

        # TODO fix the sorting of 'pairs' so I don't need to do this silly second perpetuity loop
        while True:
            for pair in pairs:
                a, b = pair
                eq = I[a,col_ix] == I[b,col_ix]
                pairwise = (p[a], p[b])
                print(f"pair: {pair} -> {pairwise}")
                match (pairwise):

                    case (True, True) | (False, False):
                        if not eq:
                            # Found a contradiction
                            print(pair)
                            return False

                    case (True, False) | (False, True):
                        if eq:
                            # Found a contradiction
                            print(pair)
                            return False

                    case (None, None):
                        continue

                    case (_, None):
                        p[b] = st(a,eq)

                    case (None, _):
                        p[a] = st(b,eq)

                    case _:
                        assert False, f"Should never happen {pairwise}"

            if all(None not in v for v in [lookup(pair) for pair in pairs]):
                break

        print(f"after: {pair} -> {(p[a],p[b])}")
        print(e[0])
        print(p)
        adjacent = [t for t in he if e[1] in t and e != t and t in he]
        for adj in adjacent:
            he.remove(adj)
            next_branch.append(adj)

        try:
            he.remove(e)
        except ValueError:
            # The edges might already have been removed from the list
            pass

        try:
            e = next_branch.pop()
        except IndexError:
            # if we've been through all edges and still not found a contradiction, the hypergraph is balanced.
            assert he == [], f"{he}"
            break

    return True

def generate_hypergraphs(n, pr, edges=[2,3]):
    '''
    Genereate Shi Brzozowski type hypergraphs
    'n' is the order of H: |V|
    'pr' is the edge probability.
    '''

    assert n >= 3, "n too small"
    assert pr >= 0.0, "probability must be non-negative"
    Is = []

    k = max(edges)
    edges = [0] + edges
    weights = [1-pr if e == 0 else pr/len(edges) for e in edges]
    _1 = mu.one(n)
    rng = np.random.default_rng(time.thread_time_ns())

    for antagonism in np.arange(0.2, 0.3, 0.2):
        while True:
            c = np.array(random.choices(edges, weights=weights, k=n)) * np.sign((np.random.rand(n) - antagonism)).astype(np.int8)
            I = np.zeros((n,n))
            for ix,k in enumerate(c):
                col = np.zeros(n)
                for e in np.array(np.random.choice(range(n),size=abs(k), replace=False)):
                    col[e] = random.choices([-1,1], k=1, weights=[antagonism, 1-antagonism])[0]
                I[:,ix] = col
        
            # Remove all zero columns: no edge can have no vertices.
            cols_removed = sum(np.all(I == 0, axis=0))
            I = I[:, ~np.all(I == 0, axis=0)]

            # I have to check that it's connected.
            m = I.shape[1]
            block_adj = np.concatenate([np.block([np.zeros((n,n)),I]), np.block([I.T, np.zeros((m,m))])])
            block_L = np.diag( (block_adj @ mu.one(block_adj.shape[0])).ravel() ) - block_adj
            fielder_eigenvalue = sorted(np.linalg.eigvals(block_L))[1]

            print(cols_removed)
            if not np.isclose(fielder_eigenvalue, 0.0, atol=1e-9) and np.unique(abs(I), axis=1).shape[1] == n-cols_removed:
                # Connected and not a multigraph
                del block_adj, block_L
                break
            

        print(I)
        print(f"Balanced: {balanced(I)}")
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


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    t1 = time.perf_counter() - t0
    print(f"\n --- Total runtime: {t1:.3f}")
