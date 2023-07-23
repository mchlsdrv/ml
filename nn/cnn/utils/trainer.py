import pathlib

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm


def trainer(model, train_data_loader: torch.utils.data.DataLoader, val_data_loader: torch.utils.data.DataLoader,
            epochs: int, optimizer: torch.optim, loss_func: torch.nn, device: torch.device,
            output_dir: pathlib.Path or str):

    epoch_train_losses = np.array([])
    epoch_val_losses = np.array([])
    # - Training loop
    for epoch in tqdm(range(epochs)):
        # - Train
        btch_train_losses = np.array([])
        btch_val_losses = np.array([])
        for btch_idx, (imgs, lbls) in enumerate(train_data_loader):
            # - Store the data in CUDA
            imgs = imgs.to(device)
            lbls = lbls.to(device)

            # - Forward
            preds = model(imgs)
            loss = loss_func(preds, lbls)
            btch_train_losses = np.append(btch_train_losses, loss.item())

            # - Backward
            optimizer.zero_grad()
            loss.backward()

            # - Optimizer step
            optimizer.step()

        # - Validation
        with torch.no_grad():
            for btch_idx, (imgs, lbls) in enumerate(val_data_loader):
                imgs = imgs.to(device)
                lbls = lbls.to(device)

                preds = model(imgs)
                loss = loss_func(preds, lbls)

                btch_val_losses = np.append(btch_val_losses, loss.item())

        epoch_train_losses = np.append(epoch_train_losses, btch_train_losses.mean())
        epoch_val_losses = np.append(epoch_val_losses, btch_val_losses.mean())

        fig, ax = plt.subplots()
        ax.plot(np.arange(epoch + 1), epoch_train_losses, label='train')
        ax.plot(np.arange(epoch + 1), epoch_val_losses, label='val')
        ax.set(xlabel='Epoch', ylabel='MSE')
        fig.suptitle('Loss vs Epochs')
        plt.legend(True)
        plt.savefig(output_dir / 'loss.png')
        plt.close(fig)
