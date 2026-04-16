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
        |rho(|L|) rho(L)| (RHO), and 
        |max_svd(|M|) - max_svd(M)| (SVD).
'''

utils.set_signal()
G_TIMEOUT = 10
M_TIMEOUT = 20 * 60

def measure(I, NP=False):

    v,e = I.shape

    nipr = mu.negative_incidence_product_ratio(I)
    nir  = mu.negative_incidence_ratio(I)

    # Fiedler of bipartite incidence, indication of how connected the graph is.
    fielder_block = mu.fiedler(mu.absolute_bipartite_incidence_laplacian(I)).real

    # Shi et al.
    # No need to compute this NP-Complete problem if we already know the answer.
    mb = hga.maximum_balance(I) if 0 < nipr else 0

    # Wang et al.
    M = I.T
    L = M.T @ M
    L_abs = np.abs(M.T) @ np.abs(M)
    rho_sigma   = mu.rho(L).real
    rho_abs     = mu.rho(L_abs).real
    svd_sigma   = mu.max_svd(M).real
    svd_abs     = mu.max_svd(np.abs(M)).real

    return {"V": v, "E": e, "FB": fielder_block,"NIPR": nipr, "NIR": nir, "MB": mb, "rho_sigma": rho_sigma,
            "rho_abs": rho_abs, "svd_sigma": svd_sigma, "svd_abs": svd_abs}

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

    data = []
    for ix in range(1, a.rounds+1):
        n = random.randint(a.V[0], a.V[1]) if len(a.V) == 2 else a.V[0]
        n_cardinalities = random.randint(1, n-1)
        cardinalities = sorted(random.sample(range(2, n + 1), n_cardinalities))

        # m most be dependent on n, e.g., if n = 2, m can only be 1
        # This will follow some theorem in set theory no doubt.

        largest_number_of_possible_edges = sum(math.comb(n,c) for c in cardinalities)
        largest_card = cardinalities[-1]
        lowest_connected_hypergraph = math.ceil( (n-1) / (largest_card-1) ) # Not true for non-uniform graph, but at least 
                                                                            # it possible that a hypergraph can always generated.

        math.ceil((1/2) * n ) # Not true, but it's difficult to compute.

        m = random.randint(lowest_connected_hypergraph, largest_number_of_possible_edges)

        round_str = f"\t\tRound {ix} of {a.rounds}: |V| = {n}, |E| = {m}, cardinalities = {cardinalities}"
        try:
            t0 = time.perf_counter()
            utils.start_timer(G_TIMEOUT)
            Is = Shi_Brzozowski.generate_hypergraphs(n, m, cardinalities)
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

