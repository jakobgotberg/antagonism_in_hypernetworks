import numpy as np
import house_committees
import senate_committees
import senate_bills


R = 'Republican'
D = 'Democrat' 
d_map = {D: 1,R:-1}
r_map = {R: 1,D:-1}
n_map = {R: 1,D: 1}

def wang_degree_balance(L, allowed_error = 0.01):
    '''
    Wang et al. (2021) Theorem 3.5.
    G is balanced if the maximum single value is an eigenvalue of G's Laplacian.
    G must be connected.
    '''
    assert L.shape[0] == L.shape[1], "L is not a square matrix"
    rho = sorted(np.linalg.svdvals( np.abs(L) ))[-1]
    eigs = sorted(np.linalg.eigvals(L))[-1]
    close = True if any(np.abs(rho - e) <= allowed_error for e in eigs) else False
    print(f"close: {rho - eigs[-1]:.3f}" if close else f"not close: {rho - eigs[-1]}")

def wang_laplacian(dataset:str, reduced=False):
    assert dataset == "house committees" or dataset == "senate committees" or dataset == "senate bills"
    M = incidence(dataset, reduced, wang_incidence=True)
    return M.T @ M


def wang_incidence(dataset:str, reduced=False):
    assert dataset == "house committees" or dataset == "senate committees" or dataset == "senate bills"
    M = incidence(dataset, reduced, wang_incidence=True)
    return M


def incidence(dataset:str, reduced=False, wang_incidence=False):
    '''
    Return incidence matrix of signed hypergraph, |V| x |E|.
    '''
    assert dataset == "house committees" or dataset == "senate committees" or dataset == "senate bills"
     
    data = house_committees if dataset == "house committees" else (senate_committees if dataset == "senate committees" else senate_bills)
    E = sorted(data.get_edges(), key=len)
    V = data.get_nodes()
    columns = []
    n = len(V)

    T = lambda m,h,e : [m[V[i]] if i in e else 0 for i in range(1,n+1)]

    if reduced:
        # Reduced hypergraph do not allow hyperedges to be subsets of each other
        for e in E:
            if any(e.issubset(s) for s in E):
                E.remove(e)

    for ix, e in enumerate(E):
        h = [V[p] for p in e]
        w = h.count(D) - h.count(R)
        m = n_map if w == 0 else (r_map if w < 0 else d_map)
        c = T(m,h,e)
        columns.append(c)

    I = np.array(columns).astype(np.int8).T

    assert np.unique(I).tolist() == [-1,0,1]
    for ix, e in enumerate(E):
        col = np.flatnonzero(I[:,ix])
        assert len(col) == len(e)
        assert all(c+1 in e for c in col)

    return I if not wang_incidence else I.T

