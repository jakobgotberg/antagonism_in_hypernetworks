import random, argparse, time, math
import Shi_Brzozowski
import numpy as np
import pandas as pd
import hypergraph_algebra as hga
import matrix_utils as mu
import utils

'''
    Generates and measures synthetic oriented hypergraphs.

    Shi_Brzozowski.py  generates a list of incidence matrices with an 
    increasing level of antagonism.

    Measures 
        Maximum Balance (MB),
        Maximum Switching Equivalence (MSE), 
        |rho(|L|) rho(L)| (RHO), and 
        |max_svd(|M|) - max_svd(M)| (SVD).
'''

utils.set_signal()
G_TIMEOUT = 10
M_TIMEOUT = 45 * 60

def measure(I, NP=False):

    v,e = I.shape

    # If 0 < nir < 1, computing mb or nse is a waste of cpu cycles.
    nir = mu.negative_incidence_ratio(I)

    # Fiedler of bipartite incidence, indication of how connected the graph is.
    fielder_block = float(mu.fiedler(mu.absolute_bipartite_incidence_laplacian(I)))

    # Shi et al.
    mb = hga.maximum_balance(I) if 0 < nir < 1 else 0

    # Wang et al.
    M = I.T
    L = M.T @ M
    rho = float(hga.RHO(L))
    svd = float(hga.SVD(M))
    mse = hga.max_se(L) if 0 < nir < 1 else 0

    return {"V": v, "E": e, "FB": fielder_block,"NIR": nir, "MB": mb, "RHO": rho, "SVD": svd, "MSE": mse}

def assert_spectrum(I):

    M = I.T
    L = M.T @ M

    assert hga.rho_is_closest(L)
    assert hga.maxsvd_is_closest(M)
    return


def main():
    p = argparse.ArgumentParser(exit_on_error=True)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--file-name", default="hypergraph_data.csv")
    p.add_argument("--verbose",action=argparse.BooleanOptionalAction, default=False)
    a = p.parse_args()

    rounds = a.rounds
    data = []
    # n + m has to be small, as hga.max_se is in O(2^{n+m}).
    for ix in range(1,rounds+1):
        n = random.randint(6,8)
        m = random.randint( math.ceil((1/2) * n ), math.ceil(n))
        pr = 1.0
        n_types_edges = random.randint(1,n-3)
        edges = sorted(random.sample(range(3, n + 1), n_types_edges))

        utils.feedback(f"Round {ix} of {rounds}: |V| = {n}, |E| = {m}, edges = {edges}", a.verbose)
        try:
            t0 = time.perf_counter()
            utils.start_timer(G_TIMEOUT)
            Is = Shi_Brzozowski.generate_hypergraphs(n, pr, m, edges)
            utils.feedback(f"Generation: {time.perf_counter() - t0:.3f} s", a.verbose)
            utils.cancel_timer()
        except utils.TimeoutExpired:
            utils.feedback("Generation Timeout", a.verbose)
            continue

        try:
            t0 = time.perf_counter()
            utils.start_timer(M_TIMEOUT)
            for I in Is:
                assert_spectrum(I)
                data.append(measure(I))
            utils.feedback(f"Measurement: {time.perf_counter() - t0:.3f} s", a.verbose)
            utils.cancel_timer()
        except utils.TimeoutExpired:
            utils.feedback("Measurement Timeout", a.verbose)
            continue

    file_name = a.file_name
    pd.DataFrame(data).to_csv(file_name, index=False)
    df = pd.read_csv(file_name)
    corr = df[["V","E","FB","NIR", "MB", "RHO", "SVD", "MSE"]].corr()
    print("\n\n")
    print(corr)
    return


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"\nRuntime: {(time.perf_counter() - t0)/60:.1f} min")


