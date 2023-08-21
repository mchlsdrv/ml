import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import pathlib
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.style.use('ggplot')


def get_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, help='The name of the experiment')
    parser.add_argument('--debug', default=False, action='store_true', help=f'If the run is a debugging run')
    parser.add_argument('--load_weights', default=False, action='store_true', help=f'If to load existing weights')
    parser.add_argument('--train', default=False, action='store_true', help=f'If to train the model')
    parser.add_argument('--test', default=False, action='store_true', help=f'If to test the model')
    parser.add_argument('--infer', default=False, action='store_true', help=f'If to infer images with the model')
    parser.add_argument('--gpu_id', type=int, default=-1, help='The ID of the GPU to run on')
    parser.add_argument('--epochs', type=int, default=100, help='The number of epochs to train the network')
    parser.add_argument('--batch_size', type=int, default=32, help='The size of the train batch')
    parser.add_argument('--n_channels', type=int, choices=[1, 3], default=3, help='1 - Gray scale, 3 - RGB')
    parser.add_argument('--output_size', type=int, choices=[1, 2], default=2, help='Number of output neurons')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='The learning rate value for the training')

    # - GENERAL PARAMETERS
    parser.add_argument('--project', type=str, help='The name of the project')

    return parser


LOSS_PLOT_RESOLUTION = 10

def train(model, train_data_loader, val_data_loader, epochs: int, optimizer, loss_func,
          device: torch.device = torch.device('cpu'), output_dir: pathlib.Path = None):
    print(f'TRAINING MODEL ...')
    epoch_train_losses = np.array([])
    epoch_val_losses = np.array([])

    loss_plot_start_idx, loss_plot_end_idx = 0, LOSS_PLOT_RESOLUTION
    loss_plot_train_history = []
    loss_plot_val_history = []

    # - Training loop
    train_loss_np = 0.0
    epch_pbar = tqdm(range(epochs))
    for epch in epch_pbar:
        # - Train
        btch_train_losses = np.array([])
        btch_val_losses = np.array([])
        for btch_idx, (imgs, lbls) in enumerate(train_data_loader):
            # - Store the data in CUDA
            imgs = imgs.to(device)
            lbls = lbls.to(device)

            # - Forward
            preds = model(imgs).float()
            loss = loss_func(preds, lbls)
            train_loss_np = loss.item()
            btch_train_losses = np.append(btch_train_losses, train_loss_np)

            # - Backward
            optimizer.zero_grad()
            loss.backward()

            # - Optimizer step
            optimizer.step()

        # - Validation
        model.eval()
        val_loss_np = 0.0
        with torch.no_grad():
            for btch_idx, (imgs, lbls) in enumerate(val_data_loader):
                imgs = imgs.to(device)
                lbls = lbls.to(device)

                preds = model(imgs)
                loss = loss_func(preds, lbls)
                val_loss_np = loss.item()
                btch_val_losses = np.append(btch_val_losses, val_loss_np)

        model.train()
        epoch_train_losses = np.append(epoch_train_losses, btch_train_losses.mean())
        epoch_val_losses = np.append(epoch_val_losses, btch_val_losses.mean())

        if len(epoch_train_losses) >= loss_plot_end_idx and len(epoch_val_losses) >= loss_plot_end_idx:
            epch_pbar.set_postfix(epoch=epch, train_loss=f'{train_loss_np:.3f}', val_loss=f'{val_loss_np:.3f}')
            # - Add the mean history
            loss_plot_train_history.append(epoch_train_losses[loss_plot_start_idx:loss_plot_end_idx].mean())
            loss_plot_val_history.append(epoch_val_losses[loss_plot_start_idx:loss_plot_end_idx].mean())

            # - Plot the mean history
            # - If the output_dir was not provided - save the outputs in a local dir
            if output_dir is None:
                output_dir = pathlib.Path('./outputs')

            # - If the output_dir was provided, but in a str format - convert it into pathlib.Path
            elif isinstance(output_dir, str):
                output_dir = pathlib.Path(output_dir)

            # - If the output_dir does not exist - create it
            if not output_dir.is_dir():
                os.makedirs(output_dir, exist_ok=True)

            plot_loss(
                train_losses=loss_plot_train_history,
                val_losses=loss_plot_val_history,
                x_ticks=np.arange(1, (epch + 1) // LOSS_PLOT_RESOLUTION + 1) * LOSS_PLOT_RESOLUTION,
                x_label='Epochs',
                y_label='BCE',
                title='Train vs Validation Plot',
                train_loss_marker='b-', val_loss_marker='r-',
                train_loss_label='train', val_loss_label='val',
                output_dir=output_dir
            )

            # - Save model weights
            checkpoint_dir = output_dir / 'checkpoints'
            os.makedirs(checkpoint_dir, exist_ok=True)
            save_checkpoint(model=model, filename=checkpoint_dir / f'weights_epoch_{epch}.pth.tar', epoch=epch)

            loss_plot_start_idx += LOSS_PLOT_RESOLUTION
            loss_plot_end_idx += LOSS_PLOT_RESOLUTION

    print(f'TRAINING FINISHED!')


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


def get_data_loaders_from_datasets(train_dataset: torch.utils.data.Dataset, train_batch_size: int,
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
