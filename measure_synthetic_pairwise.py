import random, argparse, time, csv, os
import Erdos_Renyi_pairwise as erp
import numpy as np
import matrix_utils as mu
import utils

utils.set_signal()
G_TIMEOUT = 30
M_TIMEOUT = 25 * 60

def measure(A):
    ner = mu.negative_edge_ratio(A)
    V = A.shape[0]
    E = np.sum( np.triu((np.abs(A) == 1)) )
    H = mu.normalized_pairwise_adjacency(A)

    AC = mu.algebraic_conflict(H).real
    FI = mu.normal_FI(H)
    fielder = mu.fiedler(mu.Laplacian(np.abs(A))).real

    return {"V": V, "E": E, "F": fielder, "NER": ner, "FI": FI, "A": AC}

def main(pid):
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--V", type=int, nargs="+", default=[8,16])
    p.add_argument("--file-name", default="pairwise_data")
    p.add_argument("--verbose",action="store_true")
    p.add_argument("--cyclic",action="store_true")
    a = p.parse_args()

    assert len(a.V) == 2 or len(a.V) == 1
    if len(a.V) == 2:
        assert a.V[0] <= a.V[1]

    cyclic_str = "-cyclic" if a.cyclic else "" 
    file_name = a.file_name + cyclic_str + "-PID-" + pid + ".csv"
    with open(file_name, "a"):
        # Check that file is writable before the computation.
        pass

    rounds = a.rounds
    data = []
    for ix in range(1,rounds+1):
        n = random.randint(a.V[0], a.V[1]) if len(a.V) == 2 else a.V[0]
        p = random.random()
        round_str = f"\t\tRound {ix} of {rounds}: |V| = {n}, pr = {p if p >= 0.1 else 0.1}"

        try:
            t0 = time.perf_counter()
            utils.start_timer(G_TIMEOUT)
            As = erp.generate_graphs(n, p if p >= 0.1 else 0.1, cyclic=a.cyclic)
            utils.cancel_timer()
        except utils.TimeoutExpired:
            utils.feedback("Generation Timeout:" + round_str, a.verbose)
            continue

        try:
            t0 = time.perf_counter()
            utils.start_timer(M_TIMEOUT)
            for A in As:
                data.append(measure(A))
            utils.cancel_timer()
        except utils.TimeoutExpired:
            utils.feedback("Measurement Timeout" + round_str, a.verbose)
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

