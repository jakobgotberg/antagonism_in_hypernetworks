import time,argparse,sys,itertools,copy, random
import numpy as np
import matrix_utils as mu


def find_cycles(I):
    '''
    Go from first node in first non-zero column in I and keep going until it finds the same node, or goes through all edges.
    '''
    # Can I use DFS with hyperedges?


def balanced(I):
    '''
    A decision problem: balanced or not balanced.
    Corollary 3.1: Checking the balance of H takes linear time

    The last checking takes O(|E|^2).
    '''

    # If col_i(a) == col_i(b), a and b are in the same partition, if we find a contradiction, col_j(a) != col_j(b),
    # I is not the incidence matrix of a balanced hypergraph.
    same = []
    diff = []
    for col_ix, col in enumerate([I[:, j:j+1] for j in range(I.shape[0])]):
        for pair in itertools.combinations(np.nonzero(col)[0],2):
            eq = I[pair[0],col_ix] == I[pair[1],col_ix]
            pair = tuple(sorted(pair))
            if eq:
                if pair not in same:
                    same.append(pair)
            else:
                if pair not in diff:
                    diff.append(pair)
            
    if any(d in same for d in diff) or any(s in diff for s in same):
        return False
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

    for antagonism in np.arange(0.0, 1+0.2, 0.2):
        while True:
            c = np.array(random.choices(edges, weights=weights, k=n)) * np.sign((np.random.rand(n) - antagonism)).astype(np.int8)
            I = np.zeros((n,n))
            for ix,k in enumerate(c):
                col = np.zeros(n)
                for e in np.array(np.random.choice(range(n),size=abs(k), replace=False)):
                    col[e] = random.choices([-1,1], k=1, weights=[antagonism, 1-antagonism])[0]
                I[:,ix] = col
        
            if np.unique(abs(I), axis=1).shape[0] == n:
                # Not a multigraph
                break

        print(I)
        print(f"Balanced: {balanced(I)}")
        Is.append(I)

    assert np.isin(np.unique(Is), [-1,0,1]).all(), f"Is contains: {np.unique(Is)}"
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
