"""
This module defines the following routines used by the 'train' step:
- ``estimator_fn``: Defines the customizable estimator type and 
parameters that are used during training to produce a model recipe.
"""
from typing import Dict, Any
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader
import torch
from torchmetrics import Accuracy


# Pytorch Lightning Model
class FogModel(LightningModule):
  def __init__(self):
    super().__init__()

    # Define PyTorch model
    classes=2
    features=3
    self.model = torch.nn.Sequential(
      torch.nn.Flatten(),
      torch.nn.Linear(features, 128),
      torch.nn.ReLU(),
      torch.nn.Dropout(0.1),
      torch.nn.Linear(128, 32),
      torch.nn.ReLU(),
      torch.nn.Dropout(0.1),
      torch.nn.Linear(32, classes),
      )
    self.accuracy = Accuracy()
    
  def forward(self, x):
    x = self.model(x)
    return torch.nn.functional.log_softmax(x, dim=1)
    
  def training_step(self, batch, batch_idx):
    x, y = batch
    logits = self(x)
    loss = torch.nn.functional.nll_loss(logits, y)    
    return loss
    
  def test_step(self, batch, batch_idx):
    print_str='test'
    x, y = batch
    logits = self(x)
    loss = torch.nn.functional.nll_loss(logits, y)
    preds = torch.argmax(logits, dim=1)        
    self.accuracy(preds, y)

    # Calling self.log will surface up scalars for you in TensorBoard
    self.log(f"{print_str}_loss", loss, prog_bar=True)
    self.log(f"{print_str}_acc", self.accuracy, prog_bar=True)
    return loss
    
  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=0.001)
      
  # This will then directly be usable with Pytorch Lightning to make a super quick model
  def train_dataloader(self):
    return DataLoader(self.train_ds, batch_size=BATCH_SIZE,shuffle=False)

  def test_dataloader(self):
    return DataLoader(self.test_ds, batch_size=BATCH_SIZE,shuffle=False)

def estimator_fn(estimator_params: Dict[str, Any] = None):
    """
    Returns an *unfitted* estimator that defines ``fit()`` and ``predict()`` methods.
    The estimator's input and output signatures should be compatible with scikit-learn
    estimators.
    """
    trainer = Trainer(
    max_epochs=estimator_params['max_epochs'],
    progress_bar_refresh_rate=1,
    )
    model = FogModel()

    if estimator_params is None:
        estimator_params = {}
    return trainer