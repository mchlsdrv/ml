import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import pathlib
import matplotlib.pyplot as plt

plt.style.use('ggplot')


def get_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, help='The name of the experiment')
    parser.add_argument('--debug', default=False, action='store_true', help=f'If the run is a debugging run')
    parser.add_argument('--load_weights', default=False, action='store_true', help=f'If to load existing weights')
    parser.add_argument('--train', default=False, action='store_true', help=f'If to train the model')
    parser.add_argument('--gpu_id', type=int, default=-1, help='The ID of the GPU to run on')

    # - GENERAL PARAMETERS
    parser.add_argument('--project', type=str, help='The name of the project')

    return parser


def get_device(gpu_id: int = 0):
    n_gpus = torch.cuda.device_count()

    print(f'- Number of available GPUs: {n_gpus}\n')

    device = torch.device('cpu')
    if -1 < gpu_id < n_gpus:
        device = torch.device(f'cuda:{gpu_id}')

    print(f'- Running on {device}\n')

    return device


def save_checkpoint(model: torch.nn.Module, filename: pathlib.Path or str = 'my_checkpoint.pth.tar', epoch: int = 0):
    if epoch > 0:
        print(f'\n=> Saving checkpoint for epoch {epoch} to {filename}')
    else:
        print(f'\n=> Saving checkpoint to {filename}')

    torch.save(model.state_dict(), filename)


def load_checkpoint(model, checkpoint: pathlib.Path):
    print(f'- Loading checkpoint from {checkpoint}')
    chkpt = torch.load(checkpoint)
    model.load_state_dict(chkpt)


def get_train_val_split(data_df: pd.DataFrame, val_prop: float = .2):
    n_items = len(data_df)
    item_idxs = np.arange(n_items)
    n_val_items = int(n_items * val_prop)

    # - Randomly pick the validation items' indices
    val_idxs = np.random.choice(item_idxs, n_val_items, replace=False)

    # - Pick the items for the validation set
    val_data = data_df.loc[val_idxs, :].reset_index(drop=True)

    # - The items for training are the once which are not included in the
    # validation set
    train_data = data_df.loc[np.setdiff1d(item_idxs, val_idxs), :].reset_index(drop=True)

    return train_data, val_data


def get_data_loaders(train_dataset: torch.utils.data.Dataset, train_batch_size: int,
                     val_dataset: torch.utils.data.Dataset, val_batch_size: int,
                     n_workers: int = 4, pin_memory: bool = True):

    # - Create the train / validation dataloaders
    train_dl = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        num_workers=n_workers,
        pin_memory=pin_memory,
        shuffle=True
    )

    val_dl = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        num_workers=n_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    return train_dl, val_dl


def get_x_ticks(epoch):
    x_ticks = np.arange(1, epoch)
    if 20 < epoch < 50:
        x_ticks = np.arange(1, epoch, 5)
    elif 50 < epoch < 100:
        x_ticks = np.arange(1, epoch, 10)
    elif 100 < epoch < 1000:
        x_ticks = np.arange(1, epoch, 50)
    elif 1000 < epoch < 10000:
        x_ticks = np.arange(1, epoch, 500)

    return x_ticks


def plot_loss(train_losses, val_losses, x_ticks: np.ndarray, x_label: str, y_label: str,
              title='Train vs Validation Plot',
              train_loss_marker='bo-', val_loss_marker='r*-',
              train_loss_label='train', val_loss_label='val', output_dir: pathlib.Path or str = './outputs'):
    fig, ax = plt.subplots()
    ax.plot(x_ticks, train_losses, train_loss_marker, label=train_loss_label)
    ax.plot(x_ticks, val_losses, val_loss_marker, label=val_loss_label)
    ax.set(xlabel=x_label, ylabel=y_label, xticks=x_ticks)
    fig.suptitle(title)
    plt.legend()
    output_dir = pathlib.Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    plot_output_dir = output_dir / 'plots'
    os.makedirs(plot_output_dir, exist_ok=True)
    fig.savefig(plot_output_dir / 'loss.png')
    plt.close(fig)
