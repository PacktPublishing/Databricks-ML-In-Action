import pandas as pd
import numpy as np
from torch.utils.data import Dataset


# Needs at least these __init__, __len__, __getitem__ class methods

class FogDataset(Dataset):
  def __init__(self, sql_statement,feature_cols, target_col):
    # load data
    self.df = sql(sqlQuery=sql_statement).toPandas()
    # set label
    self.df_label=self.df[[target_col]].astype(int)
    self.df_features=self.df[feature_cols].astype(float)
    # convert to tensors
    list_of_tensors = [torch.tensor(self.df_features[x].to_numpy()) for x in measures]
    self.dataset=torch.sum(torch.stack(list_of_tensors), dim=0)
    self.label=torch.tensor(self.df_label.to_numpy().reshape(-1))
    
  # This returns the total amount of samples in your Dataset
  def __len__(self):
    return len(self.dataset)
    
  # This returns given an index the i-th sample and label
  def __getitem__(self, idx):
    return self.dataset[idx],self.label[idx]