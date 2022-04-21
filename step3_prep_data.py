#!/usr/bin/env python3
import typing as ty 
import argparse 
import gzip
import os
from collections import Counter
import pickle
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt

import numpy as np
import prody as pd
from tqdm import tqdm 
from rdkit import Chem
from rdkit.Chem import ForwardSDMolSupplier
import torch 
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

from step2_generate_conformers import MolProps, Conformer


def cli() -> argparse.Namespace:
    """
    Create command line interface for script.

    Returns
    -------
    argparse.Namespace: Contains parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, 
        help="path to dir containing SDF conformer files")
    parser.add_argument("-o", "--output", required=True, 
        help="path to output dir for train, test, and validation datasets")
    return parser.parse_args()


def read_confs_from_sdf(path: str) -> ty.List[Conformer]:
    """
    Read RDKit Molecule objects from Generator.
    
    Arguments
    ---------
    path: str

    Returns
    -------
    ty.List[Conformer]: List of Conformer objects.
    """
    if path.endswith(".sdf"): fo = open(path, "rb")
    elif path.endswith(".sdf.gz"): fo = gzip.open(path)
    else: raise ValueError(f"cannot open file {path}") 
    suppl = ForwardSDMolSupplier(fo)
    confs = []
    for mol in suppl:
        props = MolProps()
        props.read_props(mol)
        confs.append(Conformer(mol, props))
    return confs 


def files_in_dir(dir_path: str) -> ty.List[str]:
    """
    Returns list of paths to files in dir at dir_path.
    
    Arguments
    ---------
    dir_path: str
    
    Returns
    -------
    ty.List[str]: List of paths to files in dir at dir_path.
    """
    file_paths = []
    for fn in os.listdir(dir_path):
        file_path = os.path.join(dir_path, fn)
        if os.path.isfile(file_path):
            file_paths.append(file_path)
    return file_paths


@dataclass
class MomentInfo:
    """
    Source: https://github.com/TurtleTools/geometricus/blob/master/geometricus/geometricus.py
    """
    moment_function: ty.Callable[[int, int, int, np.ndarray, np.ndarray], float]
    mu_arguments: ty.List[ty.Tuple[int, int, int]]


def mu(p, q, r, coords, centroid):
    """
    Centrol moment.

    Source: https://github.com/TurtleTools/geometricus/blob/master/geometricus/geometricus.py
    """
    return np.sum(
        ((coords[:, 0] - centroid[0]) ** p)
        * ((coords[:, 1] - centroid[1]) ** q)
        * ((coords[:, 2] - centroid[2]) ** r)
    )


def O_3(mu_200, mu_020, mu_002):
    """
    Source: https://github.com/TurtleTools/geometricus/blob/master/geometricus/geometricus.py
    """
    return mu_200 + mu_020 + mu_002


def O_4(mu_200, mu_020, mu_002, mu_110, mu_101, mu_011):
    """
    Source: https://github.com/TurtleTools/geometricus/blob/master/geometricus/geometricus.py
    """
    return (
        mu_200 * mu_020 * mu_002
        + 2 * mu_110 * mu_101 * mu_011
        - mu_002 * mu_110 ** 2
        - mu_020 * mu_101 ** 2
        - mu_200 * mu_011 ** 2
    )


def O_5(mu_200, mu_020, mu_002, mu_110, mu_101, mu_011):
    """
    Source: https://github.com/TurtleTools/geometricus/blob/master/geometricus/geometricus.py
    """
    return (
        mu_200 * mu_020
        + mu_200 * mu_002
        + mu_020 * mu_002
        - mu_110 ** 2
        - mu_101 ** 2
        - mu_011 ** 2
    )


def F(
    mu_201, 
    mu_021, 
    mu_210, 
    mu_300, 
    mu_111, 
    mu_012, 
    mu_003, 
    mu_030, 
    mu_102, 
    mu_120,
):
    """
    Source: https://github.com/TurtleTools/geometricus/blob/master/geometricus/geometricus.py
    """
    return (
        mu_003 ** 2
        + 6 * mu_012 ** 2
        + 6 * mu_021 ** 2
        + mu_030 ** 2
        + 6 * mu_102 ** 2
        + 15 * mu_111 ** 2
        - 3 * mu_102 * mu_120
        + 6 * mu_120 ** 2
        - 3 * mu_021 * mu_201
        + 6 * mu_201 ** 2
        - 3 * mu_003 * (mu_021 + mu_201)
        - 3 * mu_030 * mu_210
        + 6 * mu_210 ** 2
        - 3 * mu_012 * (mu_030 + mu_210)
        - 3 * mu_102 * mu_300
        - 3 * mu_120 * mu_300
        + mu_300 ** 2
    )


class MomentType(Enum):
    """
    Source: https://github.com/TurtleTools/geometricus/blob/master/geometricus/geometricus.py
    """
    O_3 = MomentInfo(O_3, [(2, 0, 0), (0, 2, 0), (0, 0, 2)])
    O_4 = MomentInfo(O_4, [(2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)])
    O_5 = MomentInfo(O_5, [(2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)])
    F = MomentInfo(F, [(2, 0, 1), (0, 2, 1), (2, 1, 0), (3, 0, 0), (1, 1, 1), (0, 1, 2), (0, 0, 3), (0, 3, 0), (1, 0, 2), (1, 2, 0)])

    def get_moments_from_coordinates(self, mus: ty.List[float]):
        return self.value.moment_function(*mus)


def get_moments_from_coordinates(coords: np.array) -> np.array:
    """
    Calculate moment invariants from coordinates.
    
    Source: https://github.com/TurtleTools/geometricus/blob/master/geometricus/geometricus.py
    """
    moment_types = (MomentType.O_3, MomentType.O_4, MomentType.O_5, MomentType.F)
    all_moment_mu_types = set(m for moment_type in moment_types for m in moment_type.value.mu_arguments)
    centroid = np.zeros(coords.shape[1])
    for i in range(coords.shape[1]): centroid[i] = np.mean(coords[:, i])
    mus = {
        (x, y, z): mu(float(x), float(y), float(z), coords, centroid)
        for (x, y, z) in all_moment_mu_types
    }
    moments = [
        moment_type.get_moments_from_coordinates(
            [mus[m] for m in moment_type.value.mu_arguments]
        )
        for moment_type in moment_types
    ]
    return moments


def calc_moment_invs(coords: np.array, split_size=5.0) -> np.array:
    """
    Calculate radial moment invariants for molecule coordinates.
    
    Source: https://github.com/TurtleTools/geometricus/blob/master/geometricus/geometricus.py
    """
    splits = [[i] for i in range(coords.shape[0])]
    split_inds = []
    kd_tree = pd.KDTree(coords)
    for i in range(len(splits)):
        kd_tree.search(
            center=coords[splits[i][0]],
            radius=split_size
        )
        split_inds.append(kd_tree.getIndices())
    moments = np.zeros((len(split_inds), 4))
    for i, inds in enumerate(split_inds):
        moments[i] = get_moments_from_coordinates(coords[inds])

    # Log normalize moment invariants
    moments += 1E-10
    moments = np.log(moments)

    return moments


def mol_to_graph(index: int, mol: Chem.Mol) -> Data:
    """
    Create PyTorch Geometric Data graph from RDKit Molecule object with embedded conformer.

    Arguments
    ---------
    index (int): Index of molecule in dataset.
    mol (rdkit.Chem.Mol): RDKit molecule object.

    Returns
    -------
    Data: PyTorch Geometric Data graph.
    """
    cid = list(mol.GetConformers()).pop(0).GetId()
    conf = mol.GetConformer(cid)

    row, col, bond_dists = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        bond_dist = conf.GetAtomPosition(start).Distance(conf.GetAtomPosition(end))
        bond_dists += 2 * [bond_dist]
    # bond_dists = np.random.rand(len(bond_dists)) # for random bond dists instead of calculated bond dists
    bond_feats = torch.tensor(bond_dists, dtype=torch.float)
    edge_index = torch.tensor([row, col], dtype=torch.long)

    atom_moment_invs = calc_moment_invs(conf.GetPositions())
    # atom_moment_invs = np.random.rand(mol.GetNumAtoms(), 4) # for random values instead of calculated moment invariants
    atom_feats = torch.tensor(atom_moment_invs, dtype=torch.float)

    try: 
        y = torch.tensor([int(mol.GetProp("_Target"))], dtype=torch.float)
        return Data(
            x=atom_feats,
            edge_attr=bond_feats,
            edge_index=edge_index,
            y=y
        )
    except:
        print("cannot set 'y' as mol has no '_Target' prop")
        return Data(
            x=atom_feats,
            edge_attr=bond_feats,
            edge_index=edge_index,
        )


class MoleculeDataset(InMemoryDataset):
    """
    Create a Molecular graph dataset.
    """
    def __init__(self, mols: ty.List[Chem.Mol]) -> None:
        """
        Initialize a MoleculeDataset.
        
        Arguments
        ---------
        mols (ty.List[rdkit.Chem.Mol]): RDKit molecules.
        """
        super().__init__(".", None, None, None)
        data_list = []
        for index, mol in tqdm(enumerate(mols)):
            graph = mol_to_graph(index, mol)
            data_list.append(graph)
        self.data, self.slices = self.collate(data_list)


def main() -> None:
    """
    Driver code.
    """
    args = cli()
    mols = []
    for fn in tqdm(files_in_dir(args.input)): 
        for conf in tqdm(read_confs_from_sdf(fn)):
            mols.append(conf.mol)

    train, test = train_test_split(mols, test_size=0.2)
    test, val = train_test_split(test, test_size=0.5)

    train = MoleculeDataset(train)
    print(f"Train: {len(train)} samples; class balance: {Counter(train.data.y.tolist())}")
    with open(os.path.join(args.output, "train.pickle"), "wb") as train_handle: 
        pickle.dump(train, train_handle)

    test = MoleculeDataset(test)
    print(f"Test: {len(test)} samples; class balance: {Counter(test.data.y.tolist())}")
    with open(os.path.join(args.output, "test.pickle"), "wb") as test_handle: 
        pickle.dump(test, test_handle)

    val = MoleculeDataset(val)
    print(f"Validation: {len(val)} samples; class balance: {Counter(val.data.y.tolist())}")
    with open(os.path.join(args.output, "val.pickle"), "wb") as val_handle: 
        pickle.dump(val, val_handle)


if __name__ == "__main__":
    main()
    