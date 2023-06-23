# Databricks notebook source
# MAGIC %run ./../setup

# COMMAND ----------

# MAGIC %pip install torchmetrics pytorch_lightning

# COMMAND ----------

import pandas as pd
import numpy as np
from enum import Enum
from sklearn import preprocessing
from sklearn.model_selection import StratifiedGroupKFold
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score
from pytorch_lightning import LightningModule, Trainer

# COMMAND ----------

measures = ["AccV", "AccML", "AccAP"]
target_col = "StartHesitation"

# COMMAND ----------

# Custom Dataset Class
## Needs ATLEAST 3 class methods
## __init__, __len__, __getitem__

class FogDataset(Dataset):
    def __init__(self,df,feature_cols, target_col):
        self.df = df
        # set label
        self.df_label=self.df[[target_col]].astype(int)
        self.df_features=self.df[feature_cols].astype(float)
        # convert to tensors
        list_of_tensors = [torch.tensor(self.df_features[x].to_numpy()) for x in feature_cols]
        self.dataset=torch.sum(torch.stack(list_of_tensors), dim=0)
        self.label=torch.tensor(self.df_label.to_numpy().reshape(-1))

    
    # This returns the total amount of samples in your Dataset
    def __len__(self):
        return len(self.dataset)
    
    # This returns given an index the i-th sample and label
    def __getitem__(self, idx):
        return self.dataset[idx],self.label[idx]

# COMMAND ----------

# Pytorch Lightning Model
class FogModel(LightningModule):
  def __init__(self,train,test):
    super().__init__()
    self.train_ds=train
    self.val_ds=test
    self.test_ds=test
    # Define PyTorch model
    classes=2
    features=3
    self.BATCH_SIZE = 1024
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
    self.accuracy = F1Score(task='binary')
    
  def forward(self, x):
    x = self.model(x)
    return torch.nn.functional.log_softmax(x, dim=1)
    
  def training_step(self, batch, batch_idx):
    x, y = batch
    logits = self(x)
    loss = torch.nn.functional.nll_loss(logits, y)    
    return loss
  
  def validation_step(self, batch, batch_idx):
    # Here we just reuse the test_step for testing
    return self.test_step(batch, batch_idx,print_str='val')
    
  def test_step(self, batch, batch_idx, print_str='test'):
    x, y = batch
    logits = self(x)
    loss = torch.nn.functional.nll_loss(logits, y)
    preds = torch.argmax(logits, dim=1)
    print(logits)      
    self.accuracy(preds, y)

    # Calling self.log will surface up scalars for you in TensorBoard
    self.log(f"{print_str}_loss", loss, prog_bar=True)
    self.log(f"{print_str}_acc", self.accuracy, prog_bar=True)
    return loss
    
  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=0.001)
      
  # This will then directly be usable with Pytorch Lightning to make a super quick model
  def train_dataloader(self):
    return DataLoader(self.train_ds, num_workers=4, batch_size=self.BATCH_SIZE,shuffle=False)

  def test_dataloader(self):
    return DataLoader(self.test_ds, num_workers=4, batch_size=self.BATCH_SIZE,shuffle=False)
  
  def val_dataloader(self):
    return DataLoader(self.val_ds, num_workers=4,batch_size=self.BATCH_SIZE,shuffle=False)

# COMMAND ----------

traindata = sql("SELECT t.*, m.Subject FROM hive_metastore.lakehouse_in_action.parkinsons_train_tdcsfog t INNER JOIN hive_metastore.lakehouse_in_action.parkinsons_tdcsfog_metadata m ON t.id = m.Id")

traindata = traindata.toPandas()

# COMMAND ----------


sgkf = StratifiedGroupKFold(n_splits=8, random_state=416, shuffle=True)
splits = sgkf.split(X=traindata['id'], y=traindata[target_col], groups=traindata['Subject'])
for fold, (train_index, test_index) in enumerate(splits):
    print(f"--- Fold = {fold} ---")
    print(f"Training label distribution {traindata.loc[train_index].groupby([target_col]).size()/(1.0*len(train_index))*100}")
    print(f"Testing label distribution {traindata.loc[test_index].groupby([target_col]).size()/(1.0*len(test_index))*100}")



# COMMAND ----------

splits = sgkf.split(X=traindata['id'], y=traindata[target_col], groups=traindata['Subject'])
for fold, (train_index, test_index) in enumerate(splits):
    if fold == 0:
      break

# COMMAND ----------

model_traindata = traindata.loc[train_index].reset_index(drop=True)
model_testdata = traindata.loc[test_index].reset_index(drop=True)

model_traindata = FogDataset(model_traindata,feature_cols=measures,target_col=target_col)
model_testdata = FogDataset(model_testdata,feature_cols=measures,target_col=target_col)

model = FogModel(model_traindata, model_testdata)

# COMMAND ----------

# Start the Trainer
trainer = Trainer(
    max_epochs=10,
)

# COMMAND ----------

trainer.fit(model)


# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


