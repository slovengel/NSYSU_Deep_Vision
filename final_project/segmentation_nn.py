"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import models
from torch.utils.data import DataLoader

class SegmentationNN(pl.LightningModule):

    def __init__(self, num_classes=101, hparams=None):
        super().__init__()
        self.hp = hparams

        self.features = models.alexnet(pretrained=True).features

        for param in self.features.parameters():
            param.requires_grad = False

        self.conv_to101 = nn.Conv2d(256, num_classes, 1)
        self.upsample = nn.Upsample(size=(self.hp["height"], self.hp["width"]), mode='bilinear', align_corners=True)

    def forward(self, x):
        """
        Use for inference only.
        Inputs:
        - x: PyTorch input Variable
        """

        final = nn.Sequential(self.features, self.conv_to101, self.upsample)
        x = final(x)

        return x

    def training_step(self, batch, batch_idx):
        """
        The complete training loop.
        """
        x, y, _, _, _ = batch
        y_hat = self.forward(x)
        loss = self.hp["loss"](y_hat, y)
        return loss

    def train_dataloader(self):
        """
        Wrap the dataset defined.
        This is the dataloader that the Trainer fit() method uses
        """
        return DataLoader(self.hp["train_dataset"], batch_size=self.hp["batch_size"], shuffle=True)

    def configure_optimizers(self):
        """
        Define optimizers and LR schedulers.
        """
        return torch.optim.Adam(self.parameters(), lr=self.hp["lr"])

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
