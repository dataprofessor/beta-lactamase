import torch 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SUPPORTED_EDGES = [
    "SINGLE",
    "DOUBLE",
    "TRIPLE",
    "AROMATIC"
]

SUPPORTED_ATOMS = [
    "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"
]

ATOMIC_NUMBERS =  [
    6, 7, 8, 9, 15, 16, 17, 35, 53
]


MAX_MOLECULE_SIZE = 122 # can also be taken 20, i got 122 from computing the len(atoms) for each molecule

DISABLE_RDKIT_WARNINGS = True