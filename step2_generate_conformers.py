#!/usr/bin/env python3
"""
Author:         David Meijer
Description:    Generate molecular conformers from a list of SMILES.
Usage:          python3 step2_generate_conformers.py -h
"""
import typing as ty
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np 
import argparse
from collections import Counter
import os

from tqdm import tqdm

from rdkit import RDLogger, Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds 
from rdkit.ML.Cluster import Butina


def cli() -> argparse.Namespace:
    """
    Create command line interface for script.

    Arguments
    ---------
    argparse.Namespace: contains parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, 
        help="input csv with rows as 'unique_id,smiles,binary_label'")
    parser.add_argument("-o", "--output", required=True, 
        help="output directory path")
    parser.add_argument("-n", "--name", required=True, 
        help="classification task name")
    parser.add_argument("-m", "--maxmpf", required=False, type=int, default=1000, 
        help="maximum number of conformers per SDF file")
    parser.add_argument("-w", "--workers", required=False, type=int, default=1, 
        help="number of workersto use for conformer generation")
    parser.add_argument("-t", "--threshold", required=False, type=float, default=float('inf'), 
        help="RMSD threshold for conformer clustering")
    return parser.parse_args()


def mol_from_smiles(smiles: str) -> Chem.Mol:
    """
    Create Molecule object from largest molecular fragment in SMILES string.

    Arguments
    ---------
    smiles (str): SMILES string.

    Returns
    -------
    mol (Chem.Mol): RDKit Molecule object.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol == None: raise ValueError("mol is None")
    if "." in smiles:
        try: fragments = list(Chem.GetMolFrags(mol, asMols=True))
        except Exception: raise ValueError("mol fragments are None")
        else: 
            fragments.sort(key=lambda fragment: fragment.GetNumHeavyAtoms())
            mol = fragments[-1]
    return mol


def get_complexity(mol: Chem.Mol) -> int:
    """
    Calculates complexity of molecule based on number of rotatable bonds.
    
    Arguments
    ---------
    mol (Chem.Mol): RDKit Molecule object.

    Returns
    -------
    complexity (int): Calculated complexity of molecule based on number of 
        rotatable bonds.
    """
    num_rota_bonds = CalcNumRotatableBonds(mol, strict=True)
    temp_mol = Chem.RemoveHs(mol)
    num_bonds = temp_mol.GetNumBonds()
    num_atoms = temp_mol.GetNumAtoms()
    return num_rota_bonds * (abs(num_bonds - num_atoms) + 1)


class RangeDict(dict):
    """
    Custom dict for using ranges as keys.
    """
    def __getitem__(self, item: int) -> ty.Any:
        """Returns value associated to value in key range.
        
        Arguments
        ---------
        item (int): key to look up value for.
        value (Any): value for range wherin key is.
        """
        if not isinstance(item, range):
            for key in self:
                if item in key: return self[key]
            raise KeyError(item)
        else: return super().__getitem__(item)


class NumConfMethod:
    """
    Defines methods for determining number of conformers to generate for molecule.
    """
    ROTA_TO_CONFS = RangeDict({
        range(0, 6): 50,
        range(6, 8): 100,
        range(8, 11): 200
    })
    COMP_TO_CONFS = RangeDict({
        range(0, 16): 50,
        range(16, 22): 100,
        range(22, 26): 200
    })
    DEFAULT_NUM_CONFS = 300

    @classmethod
    def rotatable_bonds(cls, mol: Chem.Mol) -> int:
        """
        Returns number of conformers associated to number rotatable bonds in molecule.
        
        Arguments
        ---------
        mol (Chem.Mol): RDKit Molecule object.

        Returns
        -------
        num_confs (int): Number of conformers associated to number of rotatable 
            bonds in molecule.
        """
        mol = Chem.RemoveHs(mol)
        num_rota_bonds = CalcNumRotatableBonds(mol, strict=True)
        try: num_confs = cls.ROTA_TO_CONFS[num_rota_bonds]
        except KeyError: num_confs = cls.DEFAULT_NUM_CONFS
        return num_confs

    @classmethod 
    def complexity(cls, mol: Chem.Mol) -> int:
        """
        Returns number of conformers associated to complexity of molecule.
        
        Arguments
        ---------
        mol (Chem.Mol): RDKit Molecule object.

        Returns
        -------
        num_confs (int): Number of conformers associated to complexity of molecule.
        """
        mol = Chem.RemoveHs(mol)
        complexity = get_complexity(mol)
        try: num_confs = cls.COMP_TO_CONFS[complexity]
        except: num_confs = cls.DEFAULT_NUM_CONFS
        return num_confs


class MolProps:
    """
    Stores custom properties for RDKit Molecule object with methods for writing out in SDF files.
    """
    DEFINED_PROPS = ["name", "weight", "target", "category"]
    
    def get_defined_props(self) -> ty.List[str]:
        """
        Returns defined variables.
        
        Returns
        -------
        defined (list of str): List of allowed props to define.
        """
        return self.DEFINED_PROPS

    def __setattr__(self, prop: str, value: ty.Any) -> None:
        """
        Gatekeeps allowed props to set.
        
        Arguments
        ---------
        prop (str): Prop name as string.
        value (Any): Value associated with prop.
        """
        assert(prop in self.DEFINED_PROPS), f"Given prop '{prop}' is not defined"
        self.__dict__[prop] = value

    def set_props(self, mol: Chem.Mol) -> None:
        """
        Set molecule props.

        Arguments
        ---------
        mol (Chem.Mol): RDKit Molecule object.
        """
        for prop, value in self.__dict__.items():
            prop_string = self._encode_prop_name(prop)
            value_string = self._encode_prop_value(value)
            mol.SetProp(prop_string, value_string)

    def read_props(self, mol: Chem.Mol) -> None:
        """
        Read mol props and set as props of self.
        """
        for prop in self.DEFINED_PROPS:
            encoded_prop = self._encode_prop_name(prop)
            try: encoded_value = mol.GetProps(encoded_prop)
            except: continue
            decoded_value = self._decode_prop_value(encoded_value)
            setattr(self, prop, decoded_value)

    def _encode_prop_name(self, prop_name: str) -> str:
        """
        Encodes prop name according to Mol attr standard.

        Arguments
        ---------
        prop_name (str): Prop name as string.

        Returns
        -------
        encoded_prop (str): Encoded prop name.
        """
        return f"_{prop_name.capitalize()}"

    def _decode_prop_name(self, prop_name: str) -> str:
        """
        Decodes prop name for Molecule object.

        Arguments
        ---------
        prop_name (str): Encoded prop name.

        Returns
        -------
        decoded_prop (str): Decoded prop name.
        """
        return prop_name[1:].lower()

    def _encode_prop_value(self, value: ty.Any) -> str:
        """
        Encodes prop value to str.

        Arguments
        ---------
        value (Any): Prop value.

        Returns
        -------
        encoded_value (str): Encoded prop value.
        """
        if isinstance(value, list): return ",".join(list(map(str, value)))
        elif isinstance(value, int): return str(value)
        elif isinstance(value, float): return str(float)
        elif isinstance(value, str): return value

    def _decode_prop_value(self, value: str) -> ty.Any:
        """
        Decodes prop value from str to inferred type.

        Arguments
        ---------
        value (str): Encoded prop value.

        Returns
        -------
        decoded_value (Any): Decoded prop value.
        """
        if "," in value: values = value.split(",")
        else: values = [value]
        decoded_value = []
        for v in values: 
            if "." in v: v = float(v)
            else: 
                try: v = int(v)
                except ValueError: v = str(v)
            decoded_value.append(v)
        if len(decoded_value) == 1: decoded_value = decoded_value[0]
        return decoded_value

    def get_num_tasks(self) -> int:
        """
        Returns length of target prop.

        Returns
        -------
        num_tasks (int): Number of tasks.
        """
        try: tasks = self.target 
        except: num_tasks = 0
        else: 
            if isinstance(tasks, list): num_tasks = len(tasks)
            else: num_tasks = 1
        return num_tasks


class Conformer:
    """
    Stores RDKit Molecule object with single embedded conformer.
    """
    def __init__(
        self, mol: Chem.Mol, 
        props: ty.Optional[MolProps] = None
    ) -> None:
        """
        Initialize conformer.
        
        Arguments
        ---------
        mol (Chem.Mol): RDKit Molecule object.
        props (MolProps): Molecule props.
        """
        self.mol = mol
        self.props = props 

    def read_props_from_mol(self) -> MolProps:
        """
        Read props from mol.
        
        Returns
        -------
        props (MolProps): Molecule props.
        """
        mol_props = MolProps()
        return mol_props.read_props(self.mol)


def calculate_conf_energy(
    mol: ty.Union[Chem.Mol, Conformer], 
    cid: int, iters_energy_optimization: 
    ty.Optional[int] = 100
) -> float:
    """
    Calculates energy of conf at cid in kcal/mol.
    
    Arguments
    ---------
    mol (Chem.Mol): RDKit Molecule object.
    cid (int): Conformer ID of embedded conformer.
    iters_energy_optimization (int, optional): Number of force field 
        optimization steps.

    Returns
    -------
    energy (float): Conformer energy in kcal/mol.
    """
    if isinstance(mol, Conformer): mol = mol._mol 
    ff = AllChem.MMFFGetMoleculeForceField(
        mol=mol, confId=cid, 
        pyMMFFMolProperties=AllChem.MMFFGetMoleculeProperties(mol)
    )
    if not ff: raise TypeError("encountered None for force field")
    ff.Initialize()
    if iters_energy_optimization is not None: 
        converged = ff.Minimize(maxits=iters_energy_optimization)
        if not converged: print("warning: energy optimization did not converge")
    return ff.CalcEnergy()


class EmbeddedMol:
    """
    Wrapper for RDKit Molecule object with embedded conformers.
    """
    def __init__(
        self, 
        embedded_mol: Chem.Mol, 
        cids: ty.List[int]
    ) -> None:
        """
        Initialize embedded molecule.
        
        Arguments
        ---------
        embedded_mol (Chem.Mol): RDKit Molecule object.
        cids (List[int]): List of conformer IDs.
        """
        self._mol = embedded_mol 
        self.cids = cids 

    def cluster_embedded_conformers(
        self, 
        rmsd_threshold: ty.Optional[float] = None
    ) -> ty.List[ty.Tuple[int]]:
        """
        Butina cluster embedded conformers on RMSD threshold.

        Arguments
        ---------
        rmsd_threshold (float): Cluster conformers under on max RMSD dissimilarity.

        Returns
        -------
        rmsd_clusters (list of tuples of conformers IDs as int): List of tuples 
            where each tuple is a cluster containing conformer IDs of conformers 
            clustered together.
        """
        dmat = AllChem.GetConformerRMSMatrix(self._mol, prealigned=False)
        rmsd_threshold = 0 if not rmsd_threshold else rmsd_threshold 
        return Butina.ClusterData(
            data=dmat, 
            nPts=len(self._mol.GetConformers()), 
            distThresh=rmsd_threshold,
            isDistData=True,
            reordering=True
        )

    def filter_conformers_on_clusters(
        self, 
        rmsd_clusters: ty.List[ty.Tuple[int]], 
        iters_energy_optimization: ty.Optional[int] = None
    ) -> ty.List[int]:
        """
        Retain lowest energy conformers for embedded molecule for every conformer cluster.
        
        Arguments
        ---------
        rmsd_clusters (list of tuples of conformer IDs as int): List of tuples 
            where ach tuple is a cluster containing conformer IDs of conformers clusterd together.
        iters_energy_optimization (int, optional): Number of optimization steps.

        Returns
        -------
        weights (list of tuples as (cid: int, weight: int)): List of retained 
            cids with their weight (number of collapsed conformers).
        """
        cids, weights = [], []
        temp_mol = deepcopy(self._mol)
        temp_mol.RemoveAllConformers()
        for cluster in rmsd_clusters:
            if len(cluster) > 1:
                energies = [
                    calculate_conf_energy(
                        mol=self._mol, 
                        cid=cluster_cid, 
                        iters_energy_optimization=iters_energy_optimization
                    )
                    for cluster_cid in cluster
                ]
                cid = cluster[energies.index(min(energies))]
            else: cid = cluster[0]
            weight = len(cluster)
            temp_mol.AddConformer(self._mol.GetConformer(cid))
            cids.append(cid)
            weights.append(weight)
        self._mol = temp_mol 
        self.cids = cids 
        return list(zip(cids, weights))

    def reset_cids(self) -> None:
        """
        Resets cids.
        """
        self.cids = [cid for cid, _ in enumerate(self._mol.GetConformers())]

    def extract_conformers(
        self, 
        weights: ty.Optional[ty.List[ty.Tuple[int, int]]] = None, 
        props: ty.Optional[MolProps] = None
    ) -> ty.List[Conformer]:
        """
        Extract embedded conformers.

        Arguments
        ---------
        weights (list of tuples as (cid: int, weight: int)): List of retained
            cids with their weight (number of collapsed conformers).
        props (MolProps): Molecule props.

        Returns
        -------
        conformers (list of Conformer): Embedded conformers.
        """
        assert(len(weights) == len(self.cids)), \
            "number of weights and number of conformers are different."
        confs = []
        for cid_idx, cid in enumerate(self.cids):
            if weights:
                cid_weight, weight = weights[cid_idx]
                assert(cid == cid_weight), \
                    "unable to match conformers with weights"
            else: weight = 1
            temp_mol = deepcopy(self._mol)
            temp_mol.RemoveAllConformers()
            temp_mol.AddConformer(self._mol.GetConformer(cid))
            if props:
                temp_props = deepcopy(props)
                temp_props.weight = weight 
                conf = Conformer(temp_mol, temp_props)
            else: conf = Conformer(temp_mol)
            confs.append(conf)
        return confs 

    def get_coords(
        self, 
        remove_hs: ty.Optional[bool] = True
    ) -> np.array:
        """
        Returns stacked conformer atom coordinates.

        Arguments
        ---------
        remove_hs (bool, optional): Remove hydrogens.
        
        Returns
        -------
        coords (np.array): Stacked conformer atom coordinates.
        """
        coords = []
        if remove_hs:
            mol = deepcopy(self._mol)
            mol = Chem.RemoveHs(mol)
        else: mol = self._mol 
        for conf in mol.GetConformers():
            coords.append(conf.GetPositions())
        return np.array(coords)


class ConfsEmbedder:
    """
    Contains method for generating and storing conformers in RDKit Molecule object.
    """
    def __init__(
        self,
        num_confs: ty.Optional[int] = None,
        num_confs_method: NumConfMethod = NumConfMethod.rotatable_bonds,
        prune_rms_threshold: float = -1.0,
        use_random_coords: bool = True,
        num_threads: int = 1,
        use_exp_torsion_angle_prefs: bool = True,
        use_small_ring_torsions: bool = False,
        use_macrocycle_torsions: bool = False,
        max_iters_optimization: ty.Optional[int] = None
    ) -> None:
        """
        Initialize ConfsEmbedder.
        
        Arguments
        ---------
        num_confs (int, optional): Number of conformers to generate.
        num_confs_method (NumConfMethod, optional): Method for generating conformers.
        prune_rms_threshold (float, optional): Prune conformers on RMSD threshold.
        use_random_coords (bool, optional): Use random coordinates.
        num_threads (int, optional): Number of threads.
        use_exp_torsion_angle_prefs (bool, optional): Use experimental torsion angle prefs.
        use_small_ring_torsions (bool, optional): Use small ring torsions.
        use_macrocycle_torsions (bool, optional): Use macrocycle torsions.
        max_iters_optimization (int, optional): Number of optimization steps.
        """
        self._num_confs = num_confs
        self._num_confs_method = num_confs_method
        self._prune_rms_threshold = prune_rms_threshold
        self._use_random_coords = use_random_coords
        self._num_threads = num_threads
        self._use_exp_torsion_angle_prefs = use_exp_torsion_angle_prefs
        self._use_small_ring_torsions = use_small_ring_torsions
        self._use_macrocycle_torsions = use_macrocycle_torsions
        self._max_iters_optimization = max_iters_optimization

    def embed_conformers(self, query_mol: Chem.Mol) -> EmbeddedMol:
        """
        Embed conformers in RDKit Molecule object and return wrapped EmbeddedMol.

        Arguments
        ---------
        query_mol (RDKit.Mol): Query molecule.
        
        Returns
        -------
        embedded_mol (EmbeddedMol): Embedded molecule.
        """
        mol = deepcopy(query_mol)
        mol = Chem.AddHs(mol)
        if not self._num_confs:
            num_confs = self.determine_num_confs(
                mol=mol, 
                method=self._num_confs_method
            )
        else: num_confs = self._num_confs
        cids = AllChem.EmbedMultipleConfs(
            mol=mol,
            randomSeed=42,
            numConfs=num_confs,
            pruneRmsThresh=self._prune_rms_threshold,
            useRandomCoords=self._use_random_coords,
            numThreads=self._num_threads,
            useExpTorsionAnglePrefs=self._use_exp_torsion_angle_prefs,
            useSmallRingTorsions=self._use_small_ring_torsions,
            useMacrocycleTorsions=self._use_macrocycle_torsions
        )
        if self._max_iters_optimization and self.is_embedded(mol):
            try: self.optimize_conformers(mol)
            except Exception as err: print(err)
        return EmbeddedMol(mol, cids=[cid for cid in cids])

    def optimize_conformers(self, mol: Chem.Mol) -> None:
        """
        Optimize conformers with MMFF94 force field.

        Arguments
        ---------
        mol (RDKit.Mol): Molecule.
        """
        try: AllChem.MMFFOptimizeMoleculeConfs(
            mol=mol, 
            maxIters=self._max_iters_optimization
        )
        except Exception as err: print(err)

    def determine_num_confs(
        self, 
        mol: Chem.Mol, 
        method: NumConfMethod
    ) -> int:
        """
        Determines number of conformers to generate for RDKit Molecule object.

        Arguments
        ---------
        mol (RDKit.Mol): Query molecule.
        method (NumConfMethod): Method for determining number of conformers.

        Returns
        -------
        num_confs (int): Number of conformers to generate.
        """
        return method(mol)

    def is_embedded(self, mol: Chem.Mol) -> bool:
        """
        Checks if wrapped RDKit Molecule object has embedded conformers.

        Arguments
        ---------
        mol (RDKit.Mol): Query molecule.

        Returns
        -------
        is_embedded (bool): True if RDKit Molecule object has embedded conformers.
        """
        return len(mol.GetConformers()) > 0


class GenerateConfs:
    """
    Contains pipelines for generating molecular conformers for molecule(s).
    """
    @classmethod
    def for_mol(
        cls,
        mol: Chem.Mol,
        props: ty.Optional[MolProps] = None,
        rmsd_threshold: float = 3.0,
        iters_energy_optimization: ty.Optional[int] = None,
        num_confs: ty.Optional[int] = None,
        num_confs_method: NumConfMethod = NumConfMethod.rotatable_bonds,
        prune_rms_threshold: float = -1.0,
        use_random_coords: bool = True,
        num_threads: int = 1,
        use_exp_torsion_angle_prefs: bool = True,
        use_small_ring_torsions: bool = False,
        use_macrocycle_torsions: bool = False,
        max_iters_optimization: ty.Optional[int] = None        
    ) -> ty.List[Conformer]:
        """
        Calculates RDKit Mols with single embedded conformer each from mol.
        
        Arguments
        ---------
        mol (RDKit.Mol): Query molecule.
        props (MolProps, optional): Molecule properties.
        rmsd_threshold (float, optional): RMSD threshold.
        iters_energy_optimization (int, optional): Number of energy optimization steps.
        num_confs (int, optional): Number of conformers to generate.
        num_confs_method (NumConfMethod, optional): Method for generating conformers.
        prune_rms_threshold (float, optional): Prune conformers on RMSD threshold.
        use_random_coords (bool, optional): Use random coordinates.
        num_threads (int, optional): Number of threads.
        use_exp_torsion_angle_prefs (bool, optional): Use experimental torsion angle prefs.
        use_small_ring_torsions (bool, optional): Use small ring torsions.
        use_macrocycle_torsions (bool, optional): Use macrocycle torsions.
        max_iters_optimization (int, optional): Number of optimization steps.
        
        Returns
        -------
        conformers (ty.List[Conformer]): List of Conformer objects.
        """
        name = props.name if (props and hasattr(props, 'name')) else 'MOL'
        try:
            embedder = ConfsEmbedder(
                num_confs=num_confs,
                num_confs_method=num_confs_method,
                prune_rms_threshold=prune_rms_threshold,
                use_random_coords=use_random_coords,
                num_threads=num_threads,
                use_exp_torsion_angle_prefs=use_exp_torsion_angle_prefs,
                use_small_ring_torsions=use_small_ring_torsions,
                use_macrocycle_torsions=use_macrocycle_torsions,
                max_iters_optimization=max_iters_optimization
            )
            emb_mol = embedder.embed_conformers(query_mol=mol)
            clusters = emb_mol.cluster_embedded_conformers(rmsd_threshold)
            weights = emb_mol.filter_conformers_on_clusters(
                clusters,
                iters_energy_optimization=iters_energy_optimization
            )
            if not props:
                props = MolProps()
            confs = emb_mol.extract_conformers(weights, props)
        except Exception as err:
            print(f"{name} throws '{err}'")
            confs = []
        return confs

    @classmethod
    def for_mol_batch(
        cls,
        mols: ty.Iterable[Chem.Mol],
        props: ty.Iterable[ty.Optional[MolProps]] = None,
        workers: int = 1,
        rmsd_threshold: float = 3.0,
        iters_energy_optimization: ty.Optional[int] = None,
        num_confs: ty.Optional[int] = None,
        num_confs_method: NumConfMethod = NumConfMethod.rotatable_bonds,
        prune_rms_threshold: float = -1.0,
        use_random_coords: bool = True,
        num_threads: int = 1,
        use_exp_torsion_angle_prefs: bool = True,
        use_small_ring_torsions: bool = False,
        use_macrocycle_torsions: bool = False,
        max_iters_optimization: ty.Optional[int] = None    
    ) -> ty.Generator[Chem.Mol, None, None]:
        """
        Generates RDKit Mols with single embedded conformer each from mol.
        
        Arguments
        ---------
        mols (ty.Iterable[RDKit.Mol]): Query molecules.
        props (ty.Iterable[MolProps], optional): Molecule properties.
        workers (int, optional): Number of workers.
        rmsd_threshold (float, optional): RMSD threshold.
        iters_energy_optimization (int, optional): Number of energy optimization steps.
        num_confs (int, optional): Number of conformers to generate.
        num_confs_method (NumConfMethod, optional): Method for generating conformers.
        prune_rms_threshold (float, optional): Prune conformers on RMSD threshold.
        use_random_coords (bool, optional): Use random coordinates.
        num_threads (int, optional): Number of threads.
        use_exp_torsion_angle_prefs (bool, optional): Use experimental torsion angle prefs.
        use_small_ring_torsions (bool, optional): Use small ring torsions.
        use_macrocycle_torsions (bool, optional): Use macrocycle torsions.
        max_iters_optimization (int, optional): Number of optimization steps.
        
        Yields
        ------
        RDKit.Mol: Generated RDKit Mol.
        """
        props = props if props else [None] * len(mols)

        def _mp_conf_gen() -> ty.Generator[Conformer, None, None]:
            """
            Helper function for multiprocessing conformer generation.
            
            Yields
            ------
            Conformer: Generated Conformer.
            """
            with ProcessPoolExecutor(max_workers=workers) as exec:
                jobs = {
                    exec.submit(
                        GenerateConfs.for_mol,
                        mol,
                        props,
                        rmsd_threshold,
                        iters_energy_optimization,
                        num_confs,
                        num_confs_method,
                        prune_rms_threshold,
                        use_random_coords,
                        num_threads,
                        use_exp_torsion_angle_prefs,
                        use_small_ring_torsions,
                        use_macrocycle_torsions,
                        max_iters_optimization
                    ): (mol, props) for (mol, props) in zip(mols, props)
                }
                for future in tqdm(as_completed(jobs), total=len(mols), leave=False):
                    try: 
                        confs = future.result()
                    except Exception as err: print(f'mp conf gen throws "{err}"" for {future}')
                    else: 
                        for conf in confs:
                            yield conf

        for conf in _mp_conf_gen():
            yield conf


def is_dir_valid(dir_path: str) -> bool:
    """
    Check if dir is valid.
    
    Arguments
    ---------
    dir_path (str): Directory path.

    Returns
    -------
    bool: True if dir is valid.
    """
    try: is_valid = os.path.isdir(dir_path)
    except Exception as err: 
        print(f"checking if dir is valid throws {err} for {dir_path}")
        is_valid = False 
    return is_valid


def make_dir(dir_path: str, exist_ok: bool = True) -> ty.Union[str, None]:
    """
    Creates dir at dir_path location.
    
    Arguments
    ---------
    dir_path (str): Directory path.
    exist_ok (bool, optional): If True, don't raise an exception if the 
        directory already exists.

    Returns
    -------
    str: Directory path.
    """
    if is_dir_valid(os.path.dirname(dir_path)):
        os.makedirs(dir_path, exist_ok=exist_ok)
        return dir_path
    else: raise ValueError("could not create output dir for conformers")


class ConfWriter:
    """
    Wrapper for RDKit SDWriter enabling writing out conformers over multiple files.
    """
    def __init__(
        self, 
        save_dir: str, 
        max_mols_per_file: int = 1000, 
        compress_sdf: bool = True
    ) -> None:
        """
        Initialize ConfWriter.
        
        Arguments
        ---------
        save_dir (str): Directory path.
        max_mols_per_file (int, optional): Max number of mols per file.
        compress_sdf (bool, optional): Compress SDF.
        """
        library_name = "conformers"
        self.base_name = "conf_{acc}.sdf"
        self.dir = make_dir(os.path.join(save_dir, library_name))
        self.written_confs = 0
        self.max_mpf = max_mols_per_file
        self.compress_sdf = compress_sdf 

        self.writer = None 
        self._buffer = []
        self._files = []

    def __del__(self) -> None:
        """
        Write out final conformers upon garbage collection.
        """
        self._write_buffer()

    def write_conformer(self, conf: Conformer) -> None:
        """
        Write RDKit Molecule with single embedded conformer out to file.
        
        Arguments
        ---------
        conf (Conformer): Conformer to write.
        """
        self._buffer.append(conf)
        self.written_confs += 1

        if not self.writer:
            fn = self.base_name.format(acc=0)
            fn_path = os.path.join(self.dir, fn)
            self.writer = Chem.SDWriter(fn_path)
            self._files.append(fn_path)

        elif self.written_confs % self.max_mpf == 0:
            self._write_buffer()
            fn = self.base_name.format(acc=len(self._files))
            fn_path = os.path.join(self.dir, fn)
            self.writer = Chem.SDWriter(fn_path)
            self._files.append(fn_path)

    def _write_buffer(self) -> None:
        """
        Writes out one library SDF file with set maximum number of conformers per file.
        """
        confs = self._buffer[:self.max_mpf]
        self._buffer = self._buffer[self.max_mpf:]
        for conf in confs:
            if conf.props:
                conf.props.set_props(conf.mol)
            self.writer.write(conf.mol)
        self.writer.close()
        if self.compress_sdf:
            self._compress(self._files[-1])

    def _compress(self, file_path: str) -> None:
        """Compress SDF file."""
        os.system(f"gzip {file_path}")


def parse_records(
    task_name: str, 
    path: str
) -> ty.Tuple[ty.List[Chem.Mol], ty.List[MolProps]]:
    """
    Parses SMILES entries from input file.

    Arguments
    ---------
    task_name (str): Name of task.
    path (str): Path to input csv file with rows as 'unique_id,smiles,binary_label'.
    
    Returns
    -------
    ty.List[Chem.Mol]: Tuple of lists of RDKit Mols with their MolProps.
    """
    mols, mol_props = [], []
    with open(path, "r") as handle:
        handle.readline()
        for line in handle:
            unique_id, smiles, binary_label = line.strip().split(',')
            mol = mol_from_smiles(smiles)
            props = MolProps()
            props.name = unique_id 
            props.target = binary_label 
            props.category = task_name 
            mols.append(mol)
            mol_props.append(props)
    return mols, mol_props


def main() -> None:
    """
    Driver code.
    """
    RDLogger.DisableLog("rdApp.*")  
    args = cli()
    mols, mol_props = parse_records(task_name=args.name, path=args.input)
    print(f"Class balance for '{args.name}' task: {Counter([int(props.target) for props in mol_props])}")
    writer = ConfWriter(save_dir=args.output, max_mols_per_file=args.maxmpf, compress_sdf=False)
    for conf in GenerateConfs.for_mol_batch(mols, mol_props, workers=args.workers, rmsd_threshold=args.threshold):
        writer.write_conformer(conf)


if __name__ == "__main__":
    main()
