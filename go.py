import copy,random, argparse, time, math, sys,signal
import Shi_Brzozowski
import numpy as np
import pandas as pd
import ahorn
import hypergraph_algebra as hga
import matrix_utils as mu

class TimeoutExpired(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutExpired

def start_timer(seconds: float) -> None:
    signal.setitimer(signal.ITIMER_REAL, seconds)

def cancel_timer() -> None:
    signal.setitimer(signal.ITIMER_REAL, 0)

signal.signal(signal.SIGALRM, _timeout_handler)

def measure(I, NP=False):

    v,e = I.shape
    M = copy.deepcopy(I).T
    mn,mm = M.shape

    u = np.unique(I,return_counts=True)
    if u[0].shape[0] == 3 and (u[0] == [-1,0,1]).all():
        pe, _, ne = u[1]
        ne_r = ne / (ne + pe)
    elif u[0].shape[0] == 2 and (u[0] == [-1,1]).all():
        pe, ne = u[1]
        ne_r = ne / (ne + pe)
    elif (u[0].shape[0] == 2 and (u[0] == [-1,0]).all()) or (u[0].shape[0] == 1 and (u[0] == [-1]).all()):
        ne_r = 1
    elif (u[0].shape[0] == 2 and (u[0] == [0, 1]).all()) or (u[0].shape[0] == 1 and (u[0] == [ 1]).all()):
        ne_r = 0
    else:
        assert False, f"{u}, shape:{u[0].shape[0]}"

    # Shi et al.
    mb = hga.maximum_balance(I) if not NP else None
    # Fiedler of bipartite incidence, indication of how balanced the graph is.
    fielder_block = mu.fiedler(mu.absolute_bipartite_incidence_laplacian(I))



    # Wang et al.
    L = M.T @ M
    fielder = float(sorted(np.linalg.eigvals(np.abs(L)))[1])
    rho_diff = hga.rho_maxrho_diff(L)
    svd_diff = hga.maxsvd_maxabssvd_diff(L)
    mse = hga.max_se(L) if NP else None
    #print (f"NER: {ne_r}, MB: {mb}, DSE: {dse}, MSE: {mse}")
    del L, M
    return {"V": v, "E": e, "F": fielder, "FB": fielder_block,"NER": ne_r, "MB": mb, "RHO": rho_diff, "SVD": svd_diff, "MSE": mse}

def assert_spectrum(I):

    M = I.T
    L = M.T @ M

    assert hga.rho_is_closest(L)
    assert hga.maxsvd_is_closest(L)
    return


def main():
    p = argparse.ArgumentParser(exit_on_error=True)
    p.add_argument("--m", type=int, default=8)
    p.add_argument("--file-name", default="data.csv")
    p.add_argument("--show",action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--NP",action=argparse.BooleanOptionalAction, default=False)
    a = p.parse_args()

    rounds = a.m
    data = []
    # n + m has to be below ~15 if the program is gonna finish this month.
    # worste case is now 2^18
    for ix in range(1,rounds+1):
        n = random.randint(6,13)
        m = random.randint( math.ceil((1/2) * n ), math.ceil((3/2) * n))
        p = 1.0
        n_types_edges = random.randint(1,n-3)
        edges = sorted(random.sample(range(3, n + 1), n_types_edges))
        print(f"Round {ix} of {rounds}: |V| = {n}, |E| = {m}, edges = {edges}")
        try:
            t0 = time.perf_counter()
            start_timer(10)
            Is = Shi_Brzozowski.generate_hypergraphs(n, p, m, edges)
            print(f"Generation: {time.perf_counter() - t0:.3f} s")
            cancel_timer()
        except TimeoutExpired:
            print("Timeout")
            continue

        for I in Is:
            assert_spectrum(I)
            measure(I, NP=a.NP)

    file_name = a.file_name
    pd.DataFrame(data).to_csv(file_name, index=False)
    df = pd.read_csv(file_name)
    corr = df[["V","E","F","FB","NER", "MB", "RHO", "SVD", "MSE"]].corr() if a.NP else df[["V","E","F","FB","NER","RHO", "SVD"]].corr()
    print("\n\n")
    print(corr)


if __name__ == "__main__":
    main()


