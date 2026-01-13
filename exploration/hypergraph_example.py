import toponetx as tnx
# I think they are defining their hypergraphs as complexes of 'cells'

H = tnx.ColoredHyperGraph()
# What does the 'rank' do? Looks like we have to manually assign the hyper-level with rank.
H.add_cell([1, 2])
H.add_cell([1, 3])
H.add_cell([2, 3])
# 'sizes' refers to the the number of edges of that color
print(H)
print("Graph: 1 - 2, 1 - 3, 2 - 3")
# Do all cell automatically get the 0 rank?


# The incidence matrix shows the relationship between two classes of objects, so rank a to rank b
# Rank 0 to rank 1 should represent nodes x edges?
# Looks to be working
print("Incidence: 0,1")
IM = H.incidence_matrix(0,1).todense()
print(IM)

# Ok, I think 0-cells are considered to be the nodes, 1-cell are H edges, 2-cells are, all types of, H-edges..?
# Hence, adjacency_matrix(0,1) will give a regular adjacency matrix.
print("Regular")
A = H.adjacency_matrix(0,1).todense()
print(A)


# I'll need to add the 'rank' exclipstely to get it to understand that it is a hyperedge.
H = tnx.ColoredHyperGraph()
H.add_cell([1, 2, 3], rank=2)
H.add_cell([1, 3, 4], rank=2)

print("Hyper")
a, b = 0,2
print(f"Incidence: {a},{b}")
IM = H.incidence_matrix(a,b).todense()
print(IM)

#A = H.adjacency_matrix(0,2).todense()
#print(A)
