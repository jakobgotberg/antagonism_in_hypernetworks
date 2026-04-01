import random, argparse, time
import Erdos_Renyi_pairwise as erp
import numpy as np
import pandas as pd
import hypergraph_algebra as hga
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

    AC = float(hga.algebraic_conflict(H))
    FI = hga.normal_FI(H)
    fielder = float(mu.fiedler(mu.Laplacian(np.abs(A))))

    return {"V": V, "E": E, "F": fielder, "NER": ner, "FI": FI, "A": AC}

def main():
    p = argparse.ArgumentParser(exit_on_error=True)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--file-name", default="pairwise_data.csv")
    p.add_argument("--verbose",action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--cyclic",action=argparse.BooleanOptionalAction, default=False)
    a = p.parse_args()

    rounds = a.rounds
    data = []
    for ix in range(1,rounds+1):
        n = random.randint(8,16)
        p = random.random()
        utils.feedback(f"\nRound {ix} of {rounds}: |V| = {n}, pr = {p if p >= 0.1 else 0.1}", a.verbose)

        try:
            t0 = time.perf_counter()
            utils.start_timer(G_TIMEOUT)
            utils.feedback(f"Generation: {time.perf_counter() - t0:.3f} s", a.verbose)
            As = erp.generate_graphs(n, p if p >= 0.1 else 0.1, cyclic=a.cyclic)
            utils.cancel_timer()
        except utils.TimeoutExpired:
            utils.feedback("Generation Timeout", a.verbose)
            continue

        try:
            t0 = time.perf_counter()
            utils.start_timer(M_TIMEOUT)
            for A in As:
                data.append(measure(A))
            utils.feedback(f"Measurement: {time.perf_counter() - t0:.3f} s", a.verbose)
            utils.cancel_timer()
        except utils.TimeoutExpired:
            utils.feedback("Measurement Timeout", a.verbose)
            continue

    file_name = a.file_name
    pd.DataFrame(data).to_csv(file_name, index=False)
    df = pd.read_csv(file_name)
    corr = df[["V","E","F","NER","FI","A"]].corr()
    print("\n\n")
    print(corr)


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"\nRuntime: {(time.perf_counter() - t0)/60:.1f} min")

