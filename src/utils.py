import os
import sys 
import CONFIG
import warnings 
import pandas as pd
import deepchem as dc
from rdkit import Chem, RDLogger
from torch_geometric.utils import to_dense_adj

warnings.filterwarnings("ignore")

if CONFIG.DISABLE_RDKIT_WARNINGS:
    RDLogger.DisableLog('rdApp.*')

# TODO
# 1. Support images and other types of molecule files


def compute_max_molecule_length(df_path, column_names):
    df = pd.read_csv(df_path)
    smiles = df[column_names].tolist()
    MAX_LEN = 0
    for smile in smiles:
        molecule = Chem.MolFromSmiles(smile)
        atoms = molecule.GetAtoms()
        MAX_LEN = max(MAX_LEN, len(atoms))
    return MAX_LEN

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def smiles_to_mol(smiles_string):
    return Chem.MolFromSmiles(smiles_string)

def mol_file_to_mol(mol_file):
    return Chem.MolFromMol2File(mol_file)

def draw_molecule(mol):
    return Chem.Draw.MolToImage(mol)

def convert_mol_to_tesor_graph(mol):
    """
    Converts the molecule to a graph representation that can be fed into the model
    """
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    f = featurizer.featurize(Chem.MolToSmiles(mol))
    data = f[0].to_pyg_graph()
    data["batch_index"] = torch.ones_like(data["x"][:, 0])
    return data

