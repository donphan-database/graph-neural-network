#!/usr/bin/env python3
import argparse
import pickle 
import typing as ty
import os

import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch import nn
from torch.optim import Optimizer, lr_scheduler, Adam
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import Set2Set, NNConv
from torch.autograd import Variable

from step3_prep_data import MoleculeDataset


def cli() -> argparse.Namespace:
    """
    Create command line interface for script.

    Returns
    -------
    argparse.Namespace: Contains parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-train", "--train", required=True, 
        help="path to train dataset pickle")
    parser.add_argument("-test", "--test", required=True, 
        help="path to test dataset pickle")
    parser.add_argument("-o", "--output", required=True, 
        help="dir to store outputs")
    parser.add_argument("-e", "--epochs", type=int, required=False, default=50, 
        help="number of training epochs")
    return parser.parse_args()


def initialize_weigths(model: nn.Module) -> None:
    """
    Initialize neural net weights.
    
    Arguments
    ---------
    model (nn.Module): Neural net model.
    """
    for param in model.parameters(): 
        if param.dim() == 1: nn.init.constant_(param, 0)
        else: nn.init.xavier_normal_(param)


class MPNN(nn.Module):
    """
    Message passing neural net.
    """
    def __init__(
        self,
        num_feats_node_input,
        num_feats_node_output,
        num_feats_edge_input,
        num_feats_edge_output,
        steps,
        activation,
        aggregation
    ) -> None:
        """
        Initialize MPNN.
        
        Arguments
        ---------
        num_feats_node_input (int): Number of features for node input.
        num_feats_node_output (int): Number of features for node output.
        num_feats_edge_input (int): Number of features for edge input.
        num_feats_edge_output (int): Number of features for edge output.
        steps (int): Number of message passing steps.
        activation (nn.Module): Activation function.
        aggregation (nn.Module): Aggregation function.
        """
        super().__init__()
        self.activation = activation 
        self.droptout = nn.Dropout(0.3)
        self.steps = steps 

        self.project = nn.Sequential(
            nn.Linear(num_feats_node_input, num_feats_node_output), 
            self.activation
        )
        edge_nn = nn.Sequential(
            nn.Linear(num_feats_edge_input, num_feats_edge_output), 
            self.activation, 
            nn.Linear(num_feats_edge_output, num_feats_node_output * num_feats_node_output)
        )
        self.gnn = NNConv(num_feats_node_output, num_feats_node_output, edge_nn, aggregation)
        self.gru = nn.GRU(num_feats_node_output, num_feats_node_output)

    def forward(self, batch: Batch) -> torch.Tensor:
        out = self.project(batch.x)
        msg = out.unsqueeze(0)
        for _ in range(self.steps):
            out = self.gnn(x=out, edge_index=batch.edge_index, edge_attr=batch.edge_attr.unsqueeze(1))
            out = self.activation(out)
            out, msg = self.gru(out.unsqueeze(0), msg)
            out = out.squeeze() 
        return out


class Shapenet(nn.Module):
    """
    Shapenet neural net.
    """
    def __init__(self) -> None:
        """
        Initialize Shapenet neural net.
        """
        super().__init__()

        node_hidden_size = 32
        edge_hidden_size = 8

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.mpnn = MPNN(
            num_feats_node_input=4,
            num_feats_node_output=node_hidden_size,
            num_feats_edge_input=1,
            num_feats_edge_output=edge_hidden_size,
            steps=3,
            activation=self.activation,
            aggregation="sum"
        )
        self.readout = Set2Set(in_channels=node_hidden_size, processing_steps=1, num_layers=1)
        self.ffn = self.feed_forward(input_dim=node_hidden_size, output_dim=1, ffn_size=3)
        initialize_weigths(self)

    def feed_forward(
        self, 
        input_dim: int, 
        output_dim: int, 
        ffn_size: int
    ) -> nn.Sequential:
        """
        Create feed forward neural net of requested size.

        Arguments
        ---------
        input_dim (int): Input dimension.
        output_dim (int): Output dimension.
        ffn_size (int): Feed forward neural net size.

        Returns
        -------
        nn.Sequential: Feed forward neural net.
        """
        if not ffn_size or ffn_size < 0 or not isinstance(ffn_size, int):
            raise ValueError(f"ffn size '{ffn_size}' is undefined")
        elif ffn_size == 1: ffn = [self.dropout, nn.Linear(2 * input_dim, output_dim)]
        elif ffn_size > 1:
            ffn = [self.dropout, nn.Linear(2 * input_dim, input_dim)]
            for _ in range(ffn_size - 2):
                ffn.extend([
                    self.activation, 
                    self.dropout, 
                    nn.Linear(input_dim, input_dim)
                ])
            ffn.extend([
                self.activation, 
                self.dropout, 
                nn.Linear(input_dim, output_dim)
            ])
        return nn.Sequential(*ffn)

    def forward(self, batch: Batch) -> torch.Tensor:
        out = self.mpnn(batch=batch)
        out = self.readout(x=out, batch=batch.batch)
        out = self.ffn(out)
        return out


def train_loop(
    model: nn.Module, 
    loader: DataLoader, 
    optimizer: Optimizer, 
    criterion: ty.Callable
) -> float:
    """
    Train loop for Shapenet.

    Arguments
    ---------
    model (nn.Module): Neural net model.
    loader (DataLoader): Data loader.
    optimizer (Optimizer): Optimizer.
    criterion (ty.Callable): Loss function.

    Returns
    -------
    float: Train loss.
    """
    model.train()
    num_samples = len(loader.dataset)
    loader = tqdm(loader, leave=False)
    acc_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch).squeeze()
        loss_batch = criterion(out, batch.y)
        loss_batch.backward()
        optimizer.step()
        acc_loss += loss_batch.item()
    avg_loss = acc_loss / num_samples
    return avg_loss


def eval_loop(
    model: nn.Module, 
    loader: DataLoader, 
    criterion: ty.Callable
) -> ty.Tuple[float, float]:
    """
    Evaluation loop for Shapenet.
    
    Arguments
    ---------
    model (nn.Module): Neural net model.
    loader (DataLoader): Data loader.
    criterion (ty.Callable): Loss function.
    
    Returns
    -------
    ty.Tuple[float, float]: Test loss and accuracy.
    """
    model.eval()
    num_samples = len(loader.dataset)
    loader = tqdm(loader, leave=False)
    acc_correct, acc_loss = 0, 0.0
    for batch in loader:
        with torch.no_grad():
            out = model(batch).squeeze()

            loss_batch = criterion(out, batch.y)
            acc_loss += loss_batch.item()

            t = Variable(torch.Tensor([0.5]))
            out = (torch.sigmoid(out) > t).float()
            correct_batch = torch.sum(out == batch.y)
            acc_correct += correct_batch.item()

    acc = acc_correct / num_samples
    avg_loss = acc_loss / num_samples
    return avg_loss, acc


def save_model(
    epoch: int, 
    model: nn.Module, 
    optimizer: Optimizer, 
    save_dir: str
) -> None:
    """
    Save model.
    
    Arguments
    ---------
    epoch (int): Epoch.
    model (nn.Module): Neural net model.
    optimizer (Optimizer): Optimizer.
    save_dir (str): Save directory.
    """
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(state, os.path.join(save_dir, f"model_epoch_{epoch}.pth.tar"))


def main() -> None:
    """
    Driver code.
    """
    args = cli()
    with open(args.train, "rb") as train_handle: train = pickle.load(train_handle)
    with open(args.test, "rb") as test_handle: test = pickle.load(test_handle)

    batch_size = 128
    train_loader = DataLoader(train, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test, shuffle=True, batch_size=batch_size)

    model = Shapenet()

    lr = 0.001
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    optimizer = Adam(params=model.parameters(), lr=lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        min_lr=0.00001
    )

    epochs, train_losses, test_losses, accuracies, learning_rates = \
        [], [], [], [], []

    num_epochs = args.epochs
    for epoch in range(num_epochs):
        epoch += 1

        lr = scheduler.optimizer.param_groups[0]['lr']
        train_loss = train_loop(
            model=model, 
            loader=train_loader, 
            optimizer=optimizer, 
            criterion=criterion
        )
        test_loss, eval_acc = eval_loop(
            model=model, 
            loader=test_loader, 
            criterion=criterion
        )
        scheduler.step(test_loss)

        epochs.append(epoch)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        accuracies.append(eval_acc)
        learning_rates.append(lr)

        print(
            f"Epoch: {epoch}/{num_epochs}; "
            f"LR: {lr}; "
            f"Train loss: {round(train_loss, 2)}; "
            f"Test loss: {round(test_loss, 2)}; "
            f"Eval acc: {round(eval_acc, 2)}"
        )

        fig, ax = plt.subplots()

        twin1 = ax.twinx()
        twin2 = ax.twinx()  
        twin2.spines.right.set_position(("axes", 1.2))

        ax.plot(epochs, train_losses, "r-", label="Train loss")
        ax.plot(epochs, test_losses, "r--", label="Test loss")
        twin1.plot(epochs, accuracies, "g-", label="Test accuracy")
        twin2.plot(epochs, learning_rates, "b-", label="Learning rate")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Average loss", color="r")
        twin1.set_ylabel("Test accuracy", color="g")
        twin2.set_ylabel("Learning rate", color="b")

        ax.legend(loc='center right')
        fig.set_size_inches(10, 5)
        fig.tight_layout()
        plt.savefig(os.path.join(args.output, "training.png"), dpi=300)
        plt.clf()

        if (epoch % 10 == 0):
            save_model(epoch, model, optimizer, args.output)


if __name__ == "__main__":
    main()
    