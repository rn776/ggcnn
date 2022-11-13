import datetime
import os
import sys
import argparse
import logging

import cv2

import torch
import torch.utils.data
import torch.optim as optim

from torchsummary import summary

import tensorboardX

from utils.visualisation.gridshow import gridshow

from utils.dataset_processing import evaluation
from utils.data import get_dataset
from models import get_network
from models.common import post_process_output
import os
from torch import optim, nn, utils, Tensor
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
# logging.basicConfig(level=logging.INFO)

import os


import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

try:
    from torchmetrics.functional import accuracy
except ImportError:
    from pytorch_lightning import accuracy

import mlflow.pytorch
from mlflow import MlflowClient

def parse_args():
    parser = argparse.ArgumentParser(description='Train GG-CNN')

    # Network
    parser.add_argument('--network', type=str, default='ggcnn', help='Network Name in .models')

    # Dataset & Data & Training
    parser.add_argument('--dataset', type=str, default='cornell', help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', type=str, default='/media/rnath/2f67be5d-bc1a-4dc6-9498-4825808bbabc/cornell_object_wise_25', help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb', type=int, default=0, help='Use RGB image for training (0/1)')
    parser.add_argument('--split', type=float, default=0.9, help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split for cross validation.')
    parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')

    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=1000, help='Batches per Epoch')
    parser.add_argument('--val-batches', type=int, default=250, help='Validation Batches')

    # Logging etc.
    parser.add_argument('--vis', action='store_true', help='Visualise the training process')
    args = parser.parse_args()
    return args


class GGCNNData(pl.LightningDataModule):
    def __init__(self, dataset="cornell", depth=True, rgb=False, dataset_path="",
                 split=0.9, ds_rotate=0.0, num_workers=8, batch_size=8
                 ):
        super().__init__()
        self.use_depth = depth
        self.use_rgb = rgb
        self.dataset_path = dataset_path
        self.split = split
        self.ds_rotate = ds_rotate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = get_dataset(dataset)

    def train_dataloader(self):
        train_dataset = self.dataset(self.dataset_path, start=0.0, end=self.split, ds_rotate=self.ds_rotate,
                                     random_rotate=True, random_zoom=True,
                                     include_depth=self.use_depth, include_rgb=self.use_rgb)
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                     num_workers=self.num_workers)
        return train_data_loader

    def val_dataloader(self):
        val_dataset = self.dataset(self.dataset_path, start=self.split, end=1.0, ds_rotate=self.ds_rotate,
                                   random_rotate=True, random_zoom=True,
                                   include_depth=self.use_depth, include_rgb=self.use_rgb)
        train_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                    num_workers=self.num_workers)
        return train_data_loader

class GGCNNModel(pl.LightningModule):
    def __init__(self, model_name="ggcnn", depth=True, rgb=False, batch_size=8,
                 batches_per_epoch=1000):
        super().__init__()
        self.grasp_model = get_network(model_name)
        self.use_depth = depth
        self.use_rgb = rgb
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.input_channels = 1*depth + 3*rgb
        self.net = self.grasp_model(input_channels=self.input_channels)

    def training_step(self, batch, batch_idx):
        x, y, _, _, _ = batch
        if batch_idx >= self.batches_per_epoch:
            self.trainer.should_stop = True
        loss = self.net.compute_loss(x, y)
        train_loss = loss['loss']
        self.log("train_loss", train_loss, on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y, didx, rot, zoom_factor = batch
        self.val_loss = self.net.compute_loss(x, y)
        self.log("val_loss", self.val_loss['loss'])

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_step_loss"] for x in outputs]).mean()
        self.log("val_loss", avg_loss, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

if __name__ == '__main__':
    args = parse_args()
    data_module = GGCNNData(dataset=args.dataset, depth=args.use_depth, rgb=args.use_rgb, dataset_path=args.dataset_path,
                 split=args.split, ds_rotate=args.ds_rotate, num_workers=args.num_workers, batch_size=args.batch_size)
    model = GGCNNModel(model_name=args.network, depth=args.use_depth, rgb=args.use_rgb, batch_size=args.batch_size,
                       batches_per_epoch=args.batches_per_epoch)

    trainer = pl.Trainer()
    trainer.fit(model=model,datamodule=data_module)
