#!/usr/bin/env python3
import argparse 
import pickle
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm 
from sklearn import metrics

from step3_prep_data import MoleculeDataset
from step4_train_model import Shapenet


def cli() -> argparse.Namespace:
    """
    Create command line interface for script.

    Returns
    -------
    argparse.Namespace: Contains parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--val", required=True, 
        help="validation dataset")
    parser.add_argument("-m", "--model", required=True, 
        help="path to model")
    parser.add_argument("-o", "--output", required=True, 
        help="path to output dir")
    return parser.parse_args()


def load_model(path: str) -> nn.Module:
    """
    Load Shapenet model.
    """
    model = Shapenet()
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"loaded Shapenet model trained for {checkpoint['epoch']} epochs")
    return model


def evaluate(
    model: nn.Module, 
    loader: DataLoader, 
    save_dir: str
) -> None:
    """
    Create ROC plot for results.
    
    Arguments
    ---------
    model: nn.Module
    loader: DataLoader
    save_dir: str
    """
    model.eval()

    loader = tqdm(loader, leave=False)
    pred_y, true_y = [], []
    for batch in loader:
        with torch.no_grad():
            out = model(batch).squeeze()
            pred = torch.sigmoid(out)
            pred_y.extend(pred.tolist())
            true_y.extend(batch.y.tolist())
    pred_y, true_y = np.array(pred_y), np.array(true_y)

    # Get predictions for correctly and wrongly predicted samples
    binary_pred_y = np.array(pred_y > 0.5, dtype=float)
    correct = pred_y[binary_pred_y == true_y]
    wrong = pred_y[binary_pred_y != true_y]

    # Create histogram of distribution predictions
    bins = bins=[i/100 for i in range(100)]
    plt.hist(
        [correct, wrong], 
        bins, 
        stacked=True, 
        density=False, 
        color=["g", "r"]
    )
    plt.xlabel("Prediction")
    plt.ylabel("Count")
    plt.legend(handles=[
        Patch(facecolor="g", edgecolor="g", label="Correct"),
        Patch(facecolor="r", edgecolor="r", label="Wrong")
    ])
    plt.axvline(x=0.5, color="k", linestyle=":")
    plt.savefig(os.path.join(save_dir, "pred_hist.png"), dpi=300)
    plt.clf()

    # Make ROC curve
    fpr, tpr, _ = metrics.roc_curve(true_y,  pred_y)
    auc = metrics.roc_auc_score(true_y, pred_y)
    plt.plot(fpr, tpr, color="k", label="AUC="+str(round(auc, 2)))
    plt.plot([0, 1], [0, 1], color='r', linestyle=":", label="x=y")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend(loc=4)
    plt.savefig(os.path.join(save_dir, "auc_curve.png"), dpi=300)
    plt.clf()


def main() -> None:
    """
    Driver code.
    """
    args = cli()
    with open(args.val, "rb") as val_handle: val = pickle.load(val_handle)
    val_loader = DataLoader(val, shuffle=True, batch_size=128)
    model = load_model(args.model)
    evaluate(model, val_loader, args.output)


if __name__ == "__main__":
    main()
