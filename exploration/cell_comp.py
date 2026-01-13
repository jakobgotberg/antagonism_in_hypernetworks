import toponetx as tnx

cc = tnx.CellComplex()
cc.add_cell([1,2], rank=1)
cc.add_cell([1,3, 5], rank=2)
print(cc.incidence_matrix(0,1).todense())
