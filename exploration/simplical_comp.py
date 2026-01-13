import numpy as np
import toponetx as tnx

edge_set = [[1, 2], [1, 3]]
face_set = [[2, 3, 4], [2, 4, 5]]
SC = tnx.SimplicialComplex(edge_set + face_set)

print("Up Laplacian")
print(SC.up_laplacian_matrix(rank=0).todense())
print(SC.up_laplacian_matrix(rank=1).todense())

print("Down Laplacian")
print(SC.down_laplacian_matrix(rank=1).todense())
