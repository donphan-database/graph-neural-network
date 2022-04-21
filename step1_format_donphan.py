#!/usr/bin/env python3
"""
Author:         David Meijer
Description:    Formats samples from https://donphan-database.github.io/ dataset.
Usage:          python3 format_donphan_antibacterial.py -h
"""
import typing as ty
import argparse

import numpy as np


def cli() -> argparse.Namespace:
    """
    Create command line interface with input checks for script.

    Returns
    ---------
    argparse.Namespace: contains parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, 
        help="path to DONPHAN dataset file in csv format")
    parser.add_argument("-o", "--output", required=True, 
        help="path to output file for parsed DONPHAN dataset data in csv format")
    parser.add_argument("-p", "--positive", required=True, nargs="+", 
        help="columns used for positive binary label")
    parser.add_argument("-n", "--negative", required=True, nargs="+", 
        help="columns used for negative binary label")
    args = parser.parse_args()
    invalid_columns = [
        "mol_id", 
        "smiles", 
        "inchi", 
        "monoisotopic_mass", 
        "molecular_formula", 
        "kingdom", 
        "supeclass", 
        "class", 
        "subclass"
    ]
    if any(map(lambda s: s in invalid_columns, args.positive + args.negative)): 
        raise ValueError(f"{invalid_columns} cannot be used for labeling")
    return args


def parse_donphan(
    path: str, 
    positive: ty.List[str], 
    negative: ty.List[str], 
    out_path: str
) -> None:
    """
    Parse DONPHAN dataset file into binary labeled output file. 

    Arguments
    ---------
    path (str): Path to DONPHAN dataset file.
    positive (list of str): Columns used for labeling entries as positive.
    negative (list of str): Columns used for labeling entries as negative.
    out_path (str): Output path for parsed DONPHAN dataset file.
    """
    with open(path, 'r') as handle:
        all_column_names = handle.readline().strip().split(',')
        # Check if column names in input file are unique:
        if not len(all_column_names) == len(set(all_column_names)):
            raise ValueError("non-unique column names in input file")

        # Chec if all required column names are in input file:
        req_column_names = ["mol_id", "smiles"] + positive + negative
        for column_name in req_column_names:
            if column_name not in all_column_names:
                raise ValueError(f"'{column_name}' is not present in DONPHAN dataset file")

        # Extract indices of column names of interest in input file:
        column_inds = [column_name in req_column_names for column_name in all_column_names]

        column_names = list(np.array(all_column_names)[column_inds])
        mol_id_idx = column_names.index("mol_id")
        smiles_idx = column_names.index("smiles")
        pos_idx = [column_names.index(column_name) for column_name in positive]
        neg_idx = [column_names.index(column_name) for column_name in negative]

        total = 0
        with open(out_path, "w") as out_handle:
            out_handle.write("molecule_id,smiles,label\n")
            for line in handle:
                items = np.array(line.strip().split(","))[column_inds]

                mol_id = int(items[mol_id_idx])
                smiles = items[smiles_idx]

                pos = sum([int(items[idx]) if items[idx] != "" else 0 for idx in pos_idx])
                neg = sum([int(items[idx]) if items[idx] != "" else 0 for idx in neg_idx])
                if pos > 0 and neg > 0:
                    print(f"Skipping {mol_id}: pos count ('{pos}') and neg count ('{neg}') are both non-zero")
                    continue
                # Skip entries when they have no annotations for positive and negative classes.
                elif pos == 0 and neg == 0: 
                    continue 
                else:
                    label = 1 if pos > 0 else 0
                    out_handle.write(f"{mol_id},{smiles},{label}\n")
                    total += 1

        print(f"{total} entries written out to '{out_path}'")


def main() -> None:
    """
    Driver code.
    """
    args = cli()
    parse_donphan(
        path=args.input, 
        positive=args.positive,
        negative=args.negative,
        out_path=args.output
    )


if __name__ == "__main__":
    main()
