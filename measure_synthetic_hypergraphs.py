import random, argparse, time, math, csv, os
import statistics
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
G_TIMEOUT = 0.5 * 60
M_TIMEOUT = 5 * 60
MAX_CONSECUTIVE_ATTEMPTS = 4

def measure(I, NP=False):

    t_start = time.perf_counter()

    v,e = I.shape

    t0 = time.perf_counter()
    nipr = mu.negative_incidence_product_ratio(I)
    t_nipr = time.perf_counter() - t0


    t0 = time.perf_counter()
    nir  = mu.negative_incidence_ratio(I)
    t_nir = time.perf_counter() - t0

    # Fiedler of bipartite incidence, indication of how connected the graph is.
    t0 = time.perf_counter()
    fielder = mu.fiedler(mu.absolute_bipartite_incidence_laplacian(I)).real
    t_f = time.perf_counter() - t0

    # Shi et al.
    t0 = time.perf_counter()
    mb = hga.maximum_balance(I)
    t_mb = time.perf_counter() - t0

    # Wang et al.
    t0 = time.perf_counter()
    M = I.T
    L = M.T @ M
    L_abs = np.abs(M.T) @ np.abs(M)
    rho_sigma   = mu.rho(L).real
    rho_abs     = mu.rho(L_abs).real
    svd_sigma   = mu.max_svd(M).real
    svd_abs     = mu.max_svd(np.abs(M)).real
    t_wang = time.perf_counter() - t0
    
    t_total = time.perf_counter() - t_start


    perfs =  {"NIPR": t_nipr/t_total , "NIR": t_nir/t_total, "F": t_f/t_total,  "MB": t_mb/t_total, "wang": t_wang/t_total }

    scores =  {"V": v, "E": e, "F": fielder,"NIPR": nipr, "NIR": nir, "MB": mb, "rho_sigma": rho_sigma,
            "rho_abs": rho_abs, "svd_sigma": svd_sigma, "svd_abs": svd_abs}

    return scores, perfs

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
    p.add_argument("--file-name", default="oriented_hypergraph_data")
    p.add_argument("--verbose",action="store_true")
    p.add_argument("--chatty",action="store_true")
    a = p.parse_args()


    assert len(a.V) == 2 or len(a.V) == 1
    if len(a.V) == 2:
        assert a.V[0] <= a.V[1]

    file_name = a.file_name + "-PID-" + pid + ".csv"
    with open(file_name, "a"):
        # Check that file is writable before the computation.
        pass

    data = []
    clockings = []
    for ix in range(1, a.rounds+1):
        n = random.randint(a.V[0], a.V[1]) if len(a.V) == 2 else a.V[0]
        n_cardinalities = random.randint(1, n-1)
        cardinalities = sorted(random.sample(range(2, n + 1), n_cardinalities))

        round_str = f"\t\tRound {ix} of {a.rounds}: |V| = {n}, cardinalities = {cardinalities}"
        if a.chatty:
            print(round_str)

        try:
            utils.start_timer(G_TIMEOUT)
            Is = Shi_Brzozowski.generate_hypergraphs(n, cardinalities)
            utils.cancel_timer()
        except utils.TimeoutExpired:
            utils.feedback("Generation Timeout:" + round_str, a.verbose)
            continue

        consecutive_failed_attempts = 0
        for I in Is:
            if consecutive_failed_attempts > MAX_CONSECUTIVE_ATTEMPTS:
                utils.feedback("Measurement, Max Attempts:" + round_str, a.verbose)
                break

            assert_spectrum(I)
            try:
                utils.start_timer(M_TIMEOUT)
                measurements, perfs = measure(I)
                data.append(measurements)
                clockings.append(perfs)
                utils.cancel_timer()
                consecutive_failed_attempts = 0
            except utils.TimeoutExpired:
                utils.feedback("Measurement Timeout:" + round_str, a.verbose)
                consecutive_failed_attempts += 1
                continue

    if len(data) < 1:
        utils.feedback("No measurements", a.verbose)
        return

    for key in clockings[0]:
        metric = [row[key] for row in clockings]
        print(f" {key}\t mean: {statistics.mean(metric):.3f}\t  \
              median: {statistics.median(metric):.3f}")

    with open(file_name,"a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys() )
        writer.writeheader()
        writer.writerows(data)


if __name__ == "__main__":
    t0 = time.perf_counter()
    pid = str(os.getpid())
    main(pid)
    print(f"\n{pid} Runtime: {(time.perf_counter() - t0)/60:.1f} min")

