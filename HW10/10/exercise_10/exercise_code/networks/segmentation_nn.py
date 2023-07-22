"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import models
from torch.utils.data import DataLoader

class SegmentationNN(pl.LightningModule):

    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        self.hp = hparams
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        self.features = models.alexnet(pretrained=True).features

        for param in self.features.parameters():
            param.requires_grad = False

        self.conv_to24 = nn.Conv2d(256, num_classes, 1)
        self.upsample = nn.Upsample(size=(self.hp["height"], self.hp["width"]), mode='bilinear', align_corners = True)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        final = nn.Sequential(self.features, self.conv_to24, self.upsample)

        x = final(x)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.hp["loss"](y_hat, y)
        return loss

    def train_dataloader(self):
        return DataLoader(self.hp["train_dataset"], batch_size=self.hp["batch_size"], shuffle=True)

    def configure_optimizers(self):
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

        
class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
