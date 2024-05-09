import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from edm.qm9.bond_analyze import get_bond_order

def check_connected(positions, atom_type, dataset_info):
    adj_matrix = get_adj_matrix(positions, atom_type, dataset_info)
    mol_connected = is_fully_connected(adj_matrix)

    return mol_connected

def get_adj_matrix(positions, atom_type, dataset_info):
    atom_decoder = dataset_info['atom_decoder']
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    num_atoms = len(x)
    adj_matrix = np.zeros((num_atoms, num_atoms))

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1, atom2 = atom_decoder[atom_type[i]], atom_decoder[atom_type[j]]
            order = get_bond_order(atom1, atom2, dist)

            if order > 0:
                adj_matrix[i][j] = 1
                adj_matrix[j][i] = 1

    return adj_matrix
    
def is_fully_connected(adj_matrix):
    n = len(adj_matrix)  # Number of vertices in the graph
    visited = [False] * n
    queue = [0]  # Start BFS from vertex 0

    # BFS process
    while queue:
        vertex = queue.pop(0)
        visited[vertex] = True
        for i in range(n):
            if adj_matrix[vertex][i] == 1 and not visited[i]:
                queue.append(i)
                visited[i] = True

    # Check if all vertices were visited
    return all(visited)

