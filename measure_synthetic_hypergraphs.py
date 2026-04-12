import random, argparse, time, math, csv, os
import Shi_Brzozowski
import numpy as np
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
    fielder_block = mu.fiedler(mu.absolute_bipartite_incidence_laplacian(I)).real

    # Shi et al.
    mb = hga.maximum_balance(I) if 0 < nir < 1 else 0

    # Wang et al.
    M = I.T
    L = M.T @ M
    rho = hga.RHO(L).real
    svd = hga.SVD(M).real
    mse = hga.max_se(L) if 0 < nir < 1 else 0

    return {"V": v, "E": e, "FB": fielder_block,"NIR": nir, "MB": mb, "RHO": rho, "SVD": svd, "MSE": mse}

def assert_spectrum(I):

    M = I.T
    L = M.T @ M

    assert hga.rho_is_closest(L)
    assert hga.maxsvd_is_closest(M)
    return


def main(pid):
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--V", type=int, nargs="+", default=[8,16])
    p.add_argument("--file-name", default="hypergraph_data")
    p.add_argument("--verbose",action="store_true")
    a = p.parse_args()

    assert len(a.V) == 2 or len(a.V) == 1
    if len(a.V) == 2:
        assert a.V[0] <= a.V[1]

    file_name = a.file_name + "-PID-" + pid + ".csv"
    with open(file_name, "a"):
        # Check that file is writable before the computation.
        pass

    rounds = a.rounds
    data = []
    # n + m has to be small, as hga.max_se is in O(2^{n+m}).
    for ix in range(1,rounds+1):
        n = random.randint(a.V[0], a.V[1]) if len(a.V) == 2 else a.V[0]
        m = random.randint( math.ceil((1/2) * n ), math.ceil(n))
        pr = 1.0
        n_types_edges = random.randint(1,n-3)
        edges = sorted(random.sample(range(3, n + 1), n_types_edges))

        round_str = f"\t\tRound {ix} of {rounds}: |V| = {n}, |E| = {m}, edges = {edges}"
        try:
            t0 = time.perf_counter()
            utils.start_timer(G_TIMEOUT)
            Is = Shi_Brzozowski.generate_hypergraphs(n, pr, m, edges)
            utils.cancel_timer()
        except utils.TimeoutExpired:
            utils.feedback("Generation Timeout:" + round_str, a.verbose)
            continue

        try:
            t0 = time.perf_counter()
            utils.start_timer(M_TIMEOUT)
            for I in Is:
                assert_spectrum(I)
                data.append(measure(I))
            utils.cancel_timer()
        except utils.TimeoutExpired:
            utils.feedback("Measurement Timeout:" + round_str, a.verbose)
            continue

    if len(data) < 1:
        utils.feedback("No measurements", a.verbose)
        return

    with open(file_name,"a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys() )
        writer.writeheader()
        writer.writerows(data)


if __name__ == "__main__":
    t0 = time.perf_counter()
    pid = str(os.getpid())
    main(pid)
    print(f"\n{pid} Runtime: {(time.perf_counter() - t0)/60:.1f} min")

