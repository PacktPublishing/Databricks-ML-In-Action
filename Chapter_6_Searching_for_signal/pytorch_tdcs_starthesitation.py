# Databricks notebook source
# MAGIC %run ../global-setup $project_name=parkinsons-freezing_gait_prediction

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
from torchmetrics.classification import BinaryF1Score
from pytorch_lightning import LightningModule, Trainer
import mlflow

mlflow.pytorch.autolog()

# COMMAND ----------

measures = ["AccV", "AccML", "AccAP"]
target_col = "StartHesitation"

# COMMAND ----------

# Custom Dataset Class
## Needs ATLEAST 3 class methods
## __init__, __len__, __getitem__

class FogDataset(Dataset):
    def __init__(self, df, feature_cols, target_col):
        self.df = df
        # set label
        self.df_label = self.df[[target_col]].astype(float)
        self.df_features = self.df[feature_cols].astype(float)
        # convert to tensors
        self.dataset = torch.tensor(self.df_features.to_numpy().reshape(-1, 3), dtype=torch.float32)
        self.label = torch.tensor(self.df_label.to_numpy(), dtype=torch.float32).reshape(-1)

    # This returns the total amount of samples in your Dataset
    def __len__(self):
        return len(self.dataset)

    # This returns given an index the i-th sample and label
    def __getitem__(self, idx):
        return self.dataset[idx], self.label[idx]

# COMMAND ----------

# Pytorch Lightning Model
class FogModel(LightningModule):
  def __init__(self,train,test,val):
    super().__init__()
    self.train_ds=train
    self.val_ds=val
    self.test_ds=test
    # Define PyTorch model
    noutputs=1
    nfeatures=3
    self.model = nn.Sequential(
      nn.Flatten(),
      nn.Linear(nfeatures, 32),
      nn.ReLU(),
      nn.Dropout(0.1),
      nn.Linear(32, 32), 
      nn.ReLU(),
      nn.Dropout(0.1),
      nn.Linear(32, noutputs), 
      nn.Sigmoid()
      )
    self.F1 = BinaryF1Score()
    self.criterion = nn.BCELoss()
    
  def forward(self, x):
    x = self.model(x)
    return F.log_softmax(x, dim=1)
    
  def training_step(self, batch, batch_idx):
    x, y = batch
    target = y.unsqueeze(1)
    logits = self.model(x)
    logits = F.log_softmax(logits, dim=1)
    loss = self.criterion(logits, target)
    self.log("train_loss", loss, on_epoch=True)
    f1 = self.F1(logits, target)
    self.log(f"train_f1", f1, on_epoch=True)
    return loss
  
  def test_step(self, batch, batch_idx, print_str='test'):
    x, y = batch
    target = y.unsqueeze(1)
    logits = self.model(x)
    loss = self.criterion(logits, target)
    f1 = self.F1(logits, target)
    # Calling self.log will surface up scalars for you in TensorBoard
    self.log(f"{print_str}_loss", loss, on_epoch=True)
    self.log(f"{print_str}_f1", f1, on_epoch=True)
    return loss

  def validation_step(self, batch, batch_idx):
    # Here we just reuse the test_step for testing
    return self.test_step(batch, batch_idx,print_str='val')
    
  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=0.001)
      
  # This will then directly be usable with Pytorch Lightning to make a super quick model
  def train_dataloader(self):
    return DataLoader(self.train_ds, batch_size=2048, num_workers=4,shuffle=False)

  def test_dataloader(self):
    return DataLoader(self.test_ds, batch_size=512, num_workers=4,shuffle=False)
  
  def val_dataloader(self):
    return DataLoader(self.val_ds, batch_size=512, num_workers=4,shuffle=False)

# COMMAND ----------

traindata = sql("SELECT t.*, m.Subject FROM hive_metastore.lakehouse_in_action.parkinsons_train_tdcsfog t INNER JOIN hive_metastore.lakehouse_in_action.parkinsons_tdcsfog_metadata m ON t.id = m.Id")

traindata = traindata.toPandas()

# COMMAND ----------


sgkf = StratifiedGroupKFold(n_splits=8, random_state=416, shuffle=True)
# splits = sgkf.split(X=traindata['id'], y=traindata[target_col], groups=traindata['Subject'])
# for fold, (train_index_0, test_index) in enumerate(splits):
#     print(f"--- Fold = {fold} ---")
#     print(f"Training label distribution {traindata.loc[train_index_0].groupby([target_col]).size()/(1.0*len(train_index_0))*100}")
#     print(f"Testing label distribution {traindata.loc[test_index].groupby([target_col]).size()/(1.0*len(test_index))*100}")

# COMMAND ----------

splits = sgkf.split(X=traindata['id'], y=traindata[target_col], groups=traindata['Subject'])
for fold, (train_index_0, test_index) in enumerate(splits):
    if fold == 0:
      break

# COMMAND ----------

model_traindata = traindata.loc[train_index_0].reset_index(drop=True)
model_testdata = traindata.loc[test_index].reset_index(drop=True)

# COMMAND ----------

# valsplits = sgkf.split(X=model_traindata['id'], y=model_traindata[target_col], groups=model_traindata['Subject'])
# for fold, (train_index, val_index) in enumerate(valsplits):
#     print(f"--- Fold = {fold} ---")
#     print(f"Training label distribution {model_traindata.loc[train_index].groupby([target_col]).size()/(1.0*len(train_index))*100}")
#     print(f"Testing label distribution {model_traindata.loc[val_index].groupby([target_col]).size()/(1.0*len(val_index))*100}")

# COMMAND ----------

valsplits = sgkf.split(X=model_traindata['id'], y=model_traindata[target_col], groups=model_traindata['Subject'])
for fold, (train_index, val_index) in enumerate(valsplits):
  if fold == 3:
    break

# COMMAND ----------

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

#display(model_testdata[measures])
ss.fit(model_traindata[measures])
model_traindata[measures] = ss.transform(model_traindata[measures])
model_testdata[measures] = ss.transform(model_testdata[measures])
#display(model_testdata)

model_valdata = model_traindata.loc[val_index].reset_index(drop=True)
new_model_traindata = model_traindata.loc[train_index].reset_index(drop=True)


# COMMAND ----------

model_traindataset = FogDataset(new_model_traindata,feature_cols=measures,target_col=target_col)
model_valdataset = FogDataset(model_valdata,feature_cols=measures,target_col=target_col)
model_testdataset = FogDataset(model_testdata,feature_cols=measures,target_col=target_col)

# COMMAND ----------

# Start the Trainer
trainer = Trainer(
  max_epochs=10,
)

# COMMAND ----------

model = FogModel(model_traindataset, model_testdataset, model_valdataset)

# COMMAND ----------

trainer.fit(model)


# COMMAND ----------

trainer.validate()

# COMMAND ----------


