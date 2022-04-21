#!/usr/bin/env python3
import typing as ty 
import argparse 
from collections import Counter
import os
from tqdm import tqdm 
from rdkit import Chem 
import matplotlib.pyplot as plt
import networkx as nx


def cli() -> argparse.Namespace:
    """Create CLI for script."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="input dataset csv with SMILES column")
    parser.add_argument("-o", "--output", type=str, required=True, help="output dir for figures")
    return parser.parse_args()


def parse_smiles_column_from_input_csv(path: str) -> ty.List[str]:
    """Parse list of SMILES strings from input CSV file."""
    all_smiles, all_labels = [], []
    with open(path, "r") as handle:
        header = handle.readline().strip().split(",")
        smiles_idx = header.index("smiles")
        label_idx = header.index("label")
        for line in handle:
            line = line.strip().split(",")
            smiles = line[smiles_idx]
            all_smiles.append(smiles)
            label = int(line[label_idx])
            all_labels.append(label)
    return all_smiles, all_labels


def visualize_connectivity(output_dir: str, mols: ty.List[Chem.Mol], labels: ty.List[int]) -> None:
    """Visualize connectivity of atoms per label."""
    pos, avg_pos, neg, avg_neg = [], [], [], []
    max_pos, max_neg = [], []
    for mol, label in zip(mols, labels):
        conn = Counter()
        for bond in mol.GetBonds():
            b_idx, e_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            conn[b_idx] += 1
            conn[e_idx] += 1
        if label == 1:
            for _, v in conn.items(): pos.append(v)
            try: 
                avg_pos.append(sum(list(conn.values())) / len(conn.values()))
                max_pos.append(max(conn.values()))
            except:
                print(Chem.MolToSmiles(mol), label) 
                pass
        else: 
            for _, v in conn.items(): neg.append(v)
            try: 
                avg_neg.append(sum(list(conn.values())) / len(conn.values()))
                max_neg.append(max(conn.values()))
            except: 
                print(Chem.MolToSmiles(mol), label) 
                pass
    
    # Visualize connectivity
    plt.hist(pos, bins=[0, 1, 2, 3, 4, 5], alpha=.5, label=f"Antibacterial (N={len(pos)})")
    plt.hist(neg, bins=[0, 1, 2, 3, 4, 5], alpha=.5, label=f"Not antibacterial (N={len(neg)})")
    plt.xlabel("Number of incoming edges for atom")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "atom_connectivity.png"), dpi=300)
    plt.clf()

    # Visualize average connectivity
    bins = [i/100 for i in range(100, 300)]
    plt.hist(avg_pos, bins=bins, alpha=.5, label=f"Antibacterial (N={len(avg_pos)})")
    plt.hist(avg_neg, bins=bins, alpha=.5, label=f"Not antibacterial (N={len(avg_neg)})")
    plt.xlabel("Average number of incoming edges for atoms per molecule")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "avg_atom_connectivity.png"), dpi=300)
    plt.clf()

    # Visualize max connectivity
    bins = [i/100 for i in range(100, 300)]
    plt.hist(max_pos, bins=bins, alpha=.5, label=f"Antibacterial (N={len(max_pos)})")
    plt.hist(max_neg, bins=bins, alpha=.5, label=f"Not antibacterial (N={len(max_neg)})")
    plt.xlabel("Max number of incoming edges for atoms in molecule")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "max_atom_connectivity.png"), dpi=300)
    plt.clf()

    return avg_pos, avg_neg, max_pos, max_neg


def visualize_betweenness_centrality(output_dir: str, mols: ty.List[Chem.Mol], labels: ty.List[int]) -> None:
    """Visualize betweennes centrality for atoms in molecule per label."""
    pos_all_centr = []
    pos_avg_centr = []
    pos_max_centr = []
    neg_all_centr = []
    neg_avg_centr = []
    neg_max_centr = []
    for mol, label in tqdm(zip(mols, labels)):
        edge_list = []
        for bond in mol.GetBonds():
            edge_list.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            edge_list.append((bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()))
        graph = nx.Graph(edge_list)
        betw_centr = nx.betweenness_centrality(graph, normalized=True)
        mol_centr = []
        for atom, centr in betw_centr.items():
            mol_centr.append(centr)
            if label == 1:
                pos_all_centr.append(centr)
            else:
                neg_all_centr.append(centr)
        try:
            if label == 1:
                pos_avg_centr.append(sum(mol_centr) / len(mol_centr))
                pos_max_centr.append(max(mol_centr))
            else:
                neg_avg_centr.append(sum(mol_centr) / len(mol_centr))
                neg_max_centr.append(max(mol_centr))
        except:
            print(Chem.MolToSmiles(mol), label) 
            pass

    # Visualize connectivity
    bins = [(i + 1)/100 for i in range(100)]
    plt.hist(pos_all_centr, bins=bins, alpha=.5, label=f"Antibacterial (N={len(pos_all_centr)})")
    plt.hist(neg_all_centr, bins=bins, alpha=.5, label=f"Not antibacterial (N={len(neg_all_centr)})")
    plt.xlabel("Betweenness centrality for atom")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "atom_betweenness_centrality.png"), dpi=300)
    plt.clf()

    # Visualize average betweenness centrality
    bins = [(i + 1)/100 for i in range(100)]
    plt.hist(pos_avg_centr, bins=bins, alpha=.5, label=f"Antibacterial (N={len(pos_avg_centr)})")
    plt.hist(neg_avg_centr, bins=bins, alpha=.5, label=f"Not antibacterial (N={len(neg_avg_centr)}))")
    plt.xlabel("Average betweenness centrality for molecule")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "avg_betweenness_centrality.png"), dpi=300)
    plt.clf()

    # Visualize max betweenness centrality
    bins = [(i + 1)/100 for i in range(100)]
    plt.hist(pos_max_centr, bins=bins, alpha=.5, label=f"Antibacterial (N={len(pos_max_centr)})")
    plt.hist(neg_max_centr, bins=bins, alpha=.5, label=f"Not antibacterial (N={len(neg_max_centr)}))")
    plt.xlabel("Max betweenness centrality for molecule")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "max_betweenness_centrality.png"), dpi=300)
    plt.clf()

    return pos_avg_centr, neg_avg_centr, pos_max_centr, neg_max_centr   


def main() -> None:
    """Driver code."""
    args = cli()
    all_smiles, labels = parse_smiles_column_from_input_csv(args.input)
    mols = [Chem.MolFromSmiles(smiles) for smiles in tqdm(all_smiles)]
    print(f"failed MolFromSmiles: {sum([mol == None for mol in mols])}")
    pos_avg_conn, neg_avg_conn, pos_max_conn, neg_max_conn  = visualize_connectivity(args.output, mols, labels)
    pos_avg_centr, neg_avg_centr, pos_max_centr, neg_max_centr = visualize_betweenness_centrality(args.output, mols, labels)

    with open(os.path.join(args.output, "graph_metrics.csv"), "w") as handle:
        handle.write("avg_conn,avg_centr,label\n")
        for a, b in zip(pos_avg_conn, pos_avg_centr):
            handle.write(f"{a},{b},1\n")
        for c, d in zip(neg_avg_conn, neg_avg_centr):
            handle.write(f"{c},{d},0\n")


if __name__ == "__main__":
    main()