import copy,random
import Shi_Brzozowski
import numpy as np
import pandas as pd
import ahorn
import hypergraph_algebra as hga

def measure(I):

    u = np.unique(I,return_counts=True)
    if 1 in u[0] and -1 not in u[0]:
        ne_r = 0
    elif -1 in u[0] and 1 not in u[0]:
        ne_r = 1
    else:
        e = np.unique(I,return_counts=True)[1][2]
        ne = np.unique(I,return_counts=True)[1][0]
        ne_r = ne / (ne + e)

    M = copy.deepcopy(I).T
    mb = hga.maximum_balance(I)

    L = M.T @ M
    assert hga.wang_maxsvd_closest_to_maxeigenvalue(L)
    dse = hga.wang_degree_se(L)
    mse = hga.max_se(L)
    print (f"NER: {ne_r}, MB: {mb}, DSE: {dse}, MSE: {mse}")
    del L, M
    return {"NER": ne_r, "MB": mb, "DSE": dse, "MSE": mse}


def main():

    m = 10
    data = []
    for ix in range(1,m+1):
        print(f"Round {ix} of {m}")
        n = random.randint(5,12)
        p = random.random()
        n_edges = random.randint(1,n-3)
        edges = sorted(random.sample(range(3, n + 1), n_edges))
        Is = Shi_Brzozowski.generate_hypergraphs(n, p, edges)
        for I in Is:
            data.append(measure(I))

    pd.DataFrame(data).to_csv("data.csv", index=False)
    df = pd.read_csv("data.csv")
    corr = df[["NER", "MB", "DSE", "MSE"]].corr()
    print("\n\n")
    print(corr)

if __name__ == "__main__":
    main()


