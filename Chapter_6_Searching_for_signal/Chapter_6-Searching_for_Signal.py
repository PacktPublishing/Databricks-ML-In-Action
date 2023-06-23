# Databricks notebook source
# MAGIC %run ./../setup

# COMMAND ----------

# MAGIC %pip install torchmetrics pytorch_lightning

# COMMAND ----------

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy
from sklearn import preprocessing
from enum import Enum
import copy
from pytorch_lightning import LightningModule, Trainer
from sklearn.model_selection import StratifiedGroupKFold


# COMMAND ----------

measures = ["AccV", "AccML", "AccAP"]
labels = ["StartHesitation","Walking","Turn"]

batch_size = 1024
window_size = 32
window_future = 8
window_past = window_size - window_future
        
wx = 8
        
model_dropout = 0.2
model_hidden = 512
model_nblocks = 3
        
lr = 0.00015
num_epochs = 8

# # Perform seasonal decomposition
# seasonality_window = 10  # Adjust the window size as needed
# w = Window.orderBy(col("timestamp").cast("long")).rowsBetween(-seasonality_window, seasonality_window)
# data = data.withColumn("value_avg", avg(col("value")).over(w))
# data = data.withColumn("value_seasonal", col("value") - col("value_avg"))

# # Apply the Seasonal Hybrid ESD algorithm for event detection
# # Adjust the parameters as needed
# alpha = 0.05
# max_outliers = 5
# seasonal_period = 24  # Assuming hourly data, adjust as needed

# outliers = []
# while len(outliers) < max_outliers:
#     # Decompose the residual values
#     decomposition = seasonal_decompose(data.select("value_seasonal").rdd.map(lambda x: x[0]).collect(),
#                                        period=seasonal_period, two_sided=False)

#     # Calculate the test statistics using the Grubbs' test
#     test_statistic = grubbs.max_test_statistic(decomposition.resid, alpha)

#     # Find the index of the largest outlier in the residual values
#     outlier_index = decomposition.resid.tolist().index(max(decomposition.resid, key=abs))

#     # Add the outlier to the list
#     outliers.append(data.select("timestamp").collect()[outlier_index][0])

#     # Remove the outlier from the data
#     data = data.filter(col("timestamp") != outliers[-1])

# # Print the detected event timestamps
# for outlier in outliers:
#     print(outlier)

# # Stop the SparkSession
# spark.stop()


# COMMAND ----------

# MAGIC %scala
# MAGIC
# MAGIC from pyspark.sql.functions import col, stddev, avg
# MAGIC
# MAGIC # Read the time series data into a DataFrame
# MAGIC data = spark.read.csv("path/to/timeseries_data.csv", header=True, inferSchema=True)
# MAGIC
# MAGIC # Assuming the data has a "timestamp" column and a "value" column
# MAGIC
# MAGIC # Calculate the mean and standard deviation of the values
# MAGIC mean_value = data.select(avg(col("value"))).first()[0]
# MAGIC stddev_value = data.select(stddev(col("value"))).first()[0]
# MAGIC
# MAGIC # Define the threshold for anomaly detection
# MAGIC threshold = 3  # Adjust the threshold as needed
# MAGIC
# MAGIC # Detect anomalies based on the Z-Score
# MAGIC data = data.withColumn("z_score", (col("value") - mean_value) / stddev_value)
# MAGIC anomalies = data.filter(col("z_score").abs() > threshold)
# MAGIC
# MAGIC # Print the detected anomaly timestamps
# MAGIC anomaly_timestamps = anomalies.select("timestamp").collect()
# MAGIC for row in anomaly_timestamps:
# MAGIC     print(row[0])

# COMMAND ----------

traindata = sql("SELECT t.*, m.Subject FROM hive_metastore.lakehouse_in_action.parkinsons_train_tdcsfog t INNER JOIN hive_metastore.lakehouse_in_action.parkinsons_tdcsfog_metadata m ON t.id = m.Id")

label_count = traindata.groupBy(labels).count()

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   t.*,
# MAGIC   m.Subject
# MAGIC FROM
# MAGIC   hive_metastore.lakehouse_in_action.parkinsons_train_tdcsfog t
# MAGIC   INNER JOIN hive_metastore.lakehouse_in_action.parkinsons_tdcsfog_metadata m ON t.id = m.Id

# COMMAND ----------

traindata = traindata.toPandas()

# COMMAND ----------


sgkf = StratifiedGroupKFold(n_splits=8, random_state=416, shuffle=True)
splits = sgkf.split(X=traindata['id'], y=traindata['StartHesitation'], groups=traindata['Subject'])
for fold, (train_index, test_index) in enumerate(splits):
    print(f"--- Fold = {fold} ---")
    print(f"Training label distribution {traindata.loc[train_index].groupby(['StartHesitation']).size()/(1.0*len(train_index))*100}")
    print(f"Testing label distribution {traindata.loc[test_index].groupby(['StartHesitation']).size()/(1.0*len(test_index))*100}")



# COMMAND ----------

splits = sgkf.split(X=traindata['id'], y=traindata['StartHesitation'], groups=traindata['Subject'])
for fold, (train_index, test_index) in enumerate(splits):
    print(f"--- Fold = {fold} ---")
    print(f"Training label distribution {traindata.loc[train_index].groupby(['StartHesitation']).size()/(1.0*len(train_index))*100}")
    print(f"Testing label distribution {traindata.loc[test_index].groupby(['StartHesitation']).size()/(1.0*len(test_index))*100}")
    if fold == 0:
      break

# COMMAND ----------

tdcs_query = """
SELECT
  t.*,
  m.Subject
FROM
  hive_metastore.lakehouse_in_action.parkinsons_train_tdcsfog t
  INNER JOIN hive_metastore.lakehouse_in_action.parkinsons_tdcsfog_metadata m ON t.id = m.Id
"""

# Custom Dataset Class
## Needs ATLEAST 3 class methods
## __init__, __len__, __getitem__

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

# COMMAND ----------

tdcs = FogDataset(tdcs_query,measures,labels[0])

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

def _block(in_features, out_features, drop_rate):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features),
        nn.ReLU(),
        nn.Dropout(drop_rate)
    )

class FOGModel(nn.Module):
    def __init__(self, p=cfg.model_dropout, dim=cfg.model_hidden, nblocks=cfg.model_nblocks):
        super(FOGModel, self).__init__()
        self.dropout = nn.Dropout(p)
        self.in_layer = nn.Linear(cfg.window_size*3, dim)
        self.blocks = nn.Sequential(*[_block(dim, dim, p) for _ in range(nblocks)])
        self.out_layer = nn.Linear(dim, 3)
        
    def forward(self, x):
        x = x.view(-1, cfg.window_size*3)
        x = self.in_layer(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_layer(x)
        return x

# COMMAND ----------

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# COMMAND ----------

from torch.cuda.amp import GradScaler

def train_one_epoch(model, loader, optimizer, criterion):
    loss_sum = 0.
    scaler = GradScaler()
    
    model.train()
    for x,y,t in tqdm(loader):
        x = x.to(cfg.device).float()
        y = y.to(cfg.device).float()
        t = t.to(cfg.device).float()
        
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss = torch.mean(loss*t.unsqueeze(-1), dim=1)
        
        t_sum = torch.sum(t)
        if t_sum > 0:
            loss = torch.sum(loss)/t_sum
        else:
            loss = torch.sum(loss)*0.
        
        # loss.backward()
        scaler.scale(loss).backward()
        # optimizer.step()
        scaler.step(optimizer)
        scaler.update()
        
        optimizer.zero_grad()
        
        loss_sum += loss.item()
    
    print(f"Train Loss: {(loss_sum/len(loader)):.04f}")
    

def testation_one_epoch(model, loader, criterion):
    loss_sum = 0.
    y_true_epoch = []
    y_pred_epoch = []
    t_test_epoch = []
    
    model.eval()
    for x,y,t in tqdm(loader):
        x = x.to(cfg.device).float()
        y = y.to(cfg.device).float()
        t = t.to(cfg.device).float()
        
        with torch.no_grad():
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss = torch.mean(loss*t.unsqueeze(-1), dim=1)
            
            t_sum = torch.sum(t)
            if t_sum > 0:
                loss = torch.sum(loss)/t_sum
            else:
                loss = torch.sum(loss)*0.
        
        loss_sum += loss.item()
        y_true_epoch.append(y.cpu().numpy())
        y_pred_epoch.append(y_pred.cpu().numpy())
        t_test_epoch.append(t.cpu().numpy())
        
    y_true_epoch = np.concatenate(y_true_epoch, axis=0)
    y_pred_epoch = np.concatenate(y_pred_epoch, axis=0)
    
    t_test_epoch = np.concatenate(t_test_epoch, axis=0)
    y_true_epoch = y_true_epoch[t_test_epoch > 0, :]
    y_pred_epoch = y_pred_epoch[t_test_epoch > 0, :]
    
    scores = [average_precision_score(y_true_epoch[:,i], y_pred_epoch[:,i]) for i in range(3)]
    mean_score = np.mean(scores)
    print(f"testation Loss: {(loss_sum/len(loader)):.04f}, testation Score: {mean_score:.03f}, ClassWise: {scores[0]:.03f},{scores[1]:.03f},{scores[2]:.03f}")
    
    return mean_score

# COMMAND ----------

model = FOGModel().to(cfg.device)
print(f"Number of parameters in model - {count_parameters(model):,}")

train_dataset = FOGDataset(train_fpaths, split="train")
test_dataset = FOGDataset(test_fpaths, split="test")
print(f"lengths of datasets: train - {len(train_dataset)}, test - {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=5, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, num_workers=5)

optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
criterion = torch.nn.BCEWithLogitsLoss(reduction='none').to(cfg.device)
# sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.85)

max_score = 0.0

print("="*50)
for epoch in range(cfg.num_epochs):
    print(f"Epoch: {epoch}")
    train_one_epoch(model, train_loader, optimizer, criterion)
    score = testation_one_epoch(model, test_loader, criterion)
    # sched.step()

    if score > max_score:
        max_score = score
        torch.save(model.state_dict(), "best_model_state.h5")
        print("Saving Model ...")

    print("="*50)
    
gc.collect()

# COMMAND ----------

t_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, num_workers=5)

ids = []
preds = []

for _id, x, _ in tqdm(test_loader):
    x = x.to(cfg.device).float()
    with torch.no_grad():
        y_pred = model(x)*0.1
    
    ids.extend(_id)
    preds.extend(list(np.nan_to_num(y_pred.cpu().numpy())))

# COMMAND ----------

sample_submission = pd.read_csv("/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/sample_submission.csv")
sample_submission.shape

# COMMAND ----------

preds = np.array(preds)
submission = pd.DataFrame({'Id': ids, 'StartHesitation': np.round(preds[:,0],5), \
                           'Turn': np.round(preds[:,1],5), 'Walking': np.round(preds[:,2],5)})

submission = pd.merge(sample_submission[['Id']], submission, how='left', on='Id').fillna(0.0)
submission.to_csv("submission.csv", index=False)

# COMMAND ----------

print(submission.shape)
submission.head()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


