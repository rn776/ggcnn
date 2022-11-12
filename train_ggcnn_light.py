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
from torchvision.datasets import MNIST
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

class GGCNNModel(pl.LightningModule):
    def __init__(self, model_name="ggcnn", input_channels=1):
        super().__init__()
        self.grasp_model = get_network(model_name)
        self.net = self.grasp_model(input_channels=input_channels)

    def training_step(self, batch, batch_nb):
        x, y, _, _, _ = batch
        loss = self.net.compute_loss(x, y)
        # Use the current of PyTorch logger
        self.log("train_loss", loss['loss'], on_epoch=True)

        return loss


    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y, didx, rot, zoom_factor = batch
        x = x.view(x.size(0), -1)
        loss = self.net.compute_loss(x, y)
        self.log("val_loss", loss['loss'])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())