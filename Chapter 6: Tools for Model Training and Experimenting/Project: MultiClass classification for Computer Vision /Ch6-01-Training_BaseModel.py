# Databricks notebook source
!pip install pytorch-lightning==2.1.2 deltalake==0.14.0 deltatorch==0.0.3 evalidate==2.0.2 pillow==10.1.0
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=cv_clf

# COMMAND ----------

import torch
import torchvision
import pytorch_lightning as pl
from torch import nn, optim, Tensor, manual_seed, argmax
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics.classification import Accuracy, MulticlassConfusionMatrix
from torchvision import transforms, models
from pytorch_lightning.utilities.model_summary import ModelSummary


# COMMAND ----------

import os
import numpy as np
import io
import logging
from math import ceil
from pyspark.sql.functions import col
from PIL import Image

train_df = (spark.read.format("delta")
            .load(train_delta_path)
            ).limit(10)

display(train_df)

# COMMAND ----------

dbutils.fs.ls(volume_model_path)

# COMMAND ----------

import mlflow
from mlia_utils import mlflow_funcs

experiment_path = f"/Users/{current_user}/intel-clf-training_action"
mlflow_funcs.mlflow_set_experiment(experiment_path) 
log_path = f"{volume_file_path}/intel_image_clf/intel_training_logger_board"
!mkdir {log_path}

# COMMAND ----------

class LitCVNet(pl.LightningModule):
        # we can also define model in this Module or we can define in standard pytorch Module
        # then wrap with Pytorch-Lightning Module , You can save & load model weights without 
        # altering pytorch / Lightning module . You will learn in the later series .
        def __init__(self, num_classes = 6, learning_rate= 0.0001, family = "resnext", momentum = 0.1,  dropout_rate = 0):
            super().__init__()
            self.save_hyperparameters()
            self.family = family # we do not use familit explicitly, but you could play with 2 different family models using 1 script 
            self.momentum = momentum
            self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
            self.learning_rate = learning_rate 
            self.model = self.get_model(num_classes)
            self.loss = nn.CrossEntropyLoss()
            self.confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes)
            self.test_pred = []  # collect predictions

        def get_model(self, num_classes):
            """
            This is the function that initialises our model.
            If we wanted to use other prebuilt model libraries like timm we would put that model here
            """
            backbone = torchvision.models.wide_resnet50_2(pretrained=True)
            for param in backbone.parameters():
                param.required_grad = False
            num_ftrt = backbone.fc.in_features
            backbone.fc = nn.Linear(num_ftrt, num_classes)
            return backbone

        # We do not overwrite our forward pass 
        def forward(self, x):
            x  = self.model(x)
            return x
        
        def training_step(self, batch, batch_idx):
            x = batch["content"]
            y = batch["label_id"]
            logits = self.forward(x)
            loss = self.loss(logits,y)
            # Track accuracy
            acc = self.accuracy(logits, y)
            # required format
            self.log("loss", torch.tensor([loss]), on_step=True, on_epoch=True, logger=True)
            self.log("acc", torch.tensor([acc]), on_step=True, on_epoch=True, logger=True)
            return  {"loss": loss, "acc": acc}
        
        def validation_step(self, batch, batch_idx):
            x = batch["content"]
            y = batch["label_id"]
            logits = self.forward(x)
            loss = self.loss(logits, y)
            acc = self.accuracy(logits, y)
            # required format
            self.log("val_loss", torch.tensor([loss]), prog_bar=True, on_step=True, on_epoch=True, logger=True)
            self.log("val_acc", torch.tensor([acc]), prog_bar=True, on_step=True, on_epoch=True, logger=True)
            return {"val_loss": loss, "val_acc": acc} 
        
        def test_step(self, batch, batch_idx):
            x = batch["content"]
            y = batch["label_id"]
            # Evaluate model
            logits = self.forward(x)
            # Track loss
            loss = self.loss(logits, y)
            self.log('test_loss', loss)
            # Track accuracy
            acc = self.accuracy(logits, y)
            self.log('test_accuracy', acc)
            # Collect predictions
            self.test_pred.extend(logits.cpu().numpy())
            # Update confusion matrix
            self.confusion_matrix.update(logits, y)

        # predict_step is optional unless you are doing some predictions
        def predict_step(self, batch, batch_idx, dataloader_idx=0):
            x = batch["content"]
            y = batch["label_id"]
            preds = self.forward(x)
            return preds
        
        def configure_optimizers(self):
            params = self.model.fc.parameters()
            optimizer = torch.optim.AdamW(params, lr=self.learning_rate)
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4,6], gamma=0.06)
            # required format 
            return {"optimizer":optimizer, "lr_scheduler":lr_scheduler}


# COMMAND ----------

model = LitCVNet()
ModelSummary(model, max_depth=4)

# COMMAND ----------

from deltatorch import create_pytorch_dataloader, FieldSpec

class DeltaDataModule(pl.LightningDataModule):
    """
    Creating a Data loading module with Delta Torch loader 
    """
    def __init__(self):
        super().__init__()
        self.num_classes = 6
        # Here we are applying the same Transformation 
        # in Production case scenario you probably would like to have to separate transformers for your data
        # 1 for Train and another one for test. 
        # See the Native Torch example below. 
        
        self.transform = transforms.Compose([
                transforms.Resize((150,150)),
                transforms.RandomHorizontalFlip(p=0.5), # randomly flip and rotate
                transforms.ColorJitter(0.3,0.4,0.4,0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.425, 0.415, 0.405), (0.205, 0.205, 0.205))
                ])

    def dataloader(self, path: str, batch_size=32):
       
        return create_pytorch_dataloader(
            path,
            id_field="id",
            fields=[
                FieldSpec("content", load_image_using_pil=True, transform=self.transform),
                FieldSpec("label_id"),
            ],
            shuffle=True,
            batch_size=batch_size,
        )

    def train_dataloader(self):
        return self.dataloader(train_delta_path, batch_size=64)

    def val_dataloader(self):
        return self.dataloader(val_delta_path, batch_size=64)

    def test_dataloader(self):
        return self.dataloader(val_delta_path)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Native Torch Loader 
# MAGIC If you would like to keep your images under Volumes and load them from there here is an example for Native Loader 
# MAGIC
# MAGIC ```
# MAGIC
# MAGIC import torchvision.transforms as transforms 
# MAGIC from torchvision.transforms import ToTensor,Normalize, RandomHorizontalFlip, Resize
# MAGIC from torch.utils.data import DataLoader
# MAGIC from torch.utils.data.sampler import SubsetRandomSampler
# MAGIC from torch.autograd import Variable
# MAGIC
# MAGIC class DeltaDataModule(pl.LightningDataModule):
# MAGIC     """
# MAGIC     Creating a Data loading module with Delta Torch loader 
# MAGIC     """
# MAGIC     def __init__(self, trainining_dir, validation_dir, valid_size = 0.15):
# MAGIC         super().__init__()
# MAGIC         self.train_dir = trainining_dir
# MAGIC         self.valid_dir = validation_dir
# MAGIC         self.transform = transforms.Compose([
# MAGIC                 transforms.Resize((150,150)),
# MAGIC                 transforms.RandomHorizontalFlip(p=0.5), # randomly flip and rotate
# MAGIC                 transforms.ColorJitter(0.3,0.4,0.4,0.2),
# MAGIC                 transforms.ToTensor(),
# MAGIC                 transforms.Normalize((0.425, 0.415, 0.405), (0.205, 0.205, 0.205))
# MAGIC                 ])
# MAGIC         
# MAGIC         self.transform_tests = transforms.Compose([
# MAGIC                 transforms.Resize((150,150)),
# MAGIC                 transforms.ToTensor(),
# MAGIC                 transforms.Normalize((0.425, 0.415, 0.405), (0.255, 0.245, 0.235))
# MAGIC                 ])
# MAGIC         # ImageFloder function uses for make dataset by passing dir adderess as an argument
# MAGIC         self.train_data = torchvision.datasets.ImageFolder(root=self.train_dir, transform=self.transform)
# MAGIC         self.test_data = torchvision.datasets.ImageFolder(root=self.valid_dir, transform=self.transform_tests)
# MAGIC         self.train_sampler, self.valid_sampler = self.shuffle_data(self.train_data, valid_size = 0.15)
# MAGIC         
# MAGIC     def shuffle_data(self, train_data, valid_size = 0.15):
# MAGIC
# MAGIC         # Splot data into train and validation set
# MAGIC         num_train = len(train_data)
# MAGIC         indices = list(range(num_train))
# MAGIC         np.random.shuffle(indices)
# MAGIC
# MAGIC         split = int(np.floor(valid_size * num_train))
# MAGIC         train_idx, valid_idx = indices[split:], indices[:split]
# MAGIC
# MAGIC         train_sampler = SubsetRandomSampler(train_idx)
# MAGIC         valid_sampler = SubsetRandomSampler(valid_idx)
# MAGIC         
# MAGIC         return train_sampler, valid_sampler
# MAGIC
# MAGIC     def train_dataloader(self):
# MAGIC         return DataLoader(self.train_dir, batch_size=120, sampler=self.train_sampler, num_workers=2)
# MAGIC
# MAGIC     def val_dataloader(self):
# MAGIC         return DataLoader(self.train_dir, batch_size=50, sampler=self.valid_sampler, num_workers=2)
# MAGIC
# MAGIC     def test_dataloader(self):
# MAGIC         return DataLoader(self.test_data, batch_size=32, sampler=None, num_workers=2)
# MAGIC
# MAGIC ```

# COMMAND ----------

#Check GPU availability
if not torch.cuda.is_available(): # is gpu
  raise Exception("Please use a GPU-cluster for model training, CPU instances will be too slow")

# COMMAND ----------

MAX_EPOCH_COUNT = 30
STEPS_PER_EPOCH = 5
EARLY_STOP_MIN_DELTA = 0.05
EARLY_STOP_PATIENCE = 10
STRATEGY = "auto"

from pytorch_lightning.loggers import TensorBoardLogger

def train_distributed(max_epochs: int = 1, strategy: str = "auto"):
    # import logging
    # logging.basicConfig(level=logging.DEBUG)

    # this is required if you want to log with MlFlow
    # if you do not have access to a Token, simply save models under a checkpoint and log them back
    # Track your loss with the Tensoflow Board 
    os.environ['DATABRICKS_HOST'] = db_host
    os.environ['DATABRICKS_TOKEN'] = db_token
    torch.set_float32_matmul_precision("medium")
    #logger = TensorBoardLogger(log_path, name="cv_uc_model", default_hp_metric= True, sub_dir = "cv_uc")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=2,
        mode="min",
        monitor="acc", # this has been saved under the Model Trainer - inside the validation_step function 
        dirpath=log_path,
        filename="sample-cvops-{epoch:02d}-{val_loss:.2f}"
    )
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="loss",
        min_delta=EARLY_STOP_MIN_DELTA,
        patience=EARLY_STOP_PATIENCE,
        stopping_threshold=0.1,
        strict=True,
        verbose=True,
        mode="min",
        check_on_train_epoch_end=True,
        log_rank_zero_only=True
    )

    tqdm_callback = pl.callbacks.TQDMProgressBar(
        refresh_rate=STEPS_PER_EPOCH,
        process_position=0
    )

    # Initialize a trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        strategy=strategy,
        default_root_dir=log_path,
        max_epochs=max_epochs,
        logger=None,
        callbacks=[
            early_stop_callback,
            checkpoint_callback,
            tqdm_callback
        ],
    )

    print(f"Global Rank: {trainer.global_rank}")
    print(f"Local Rank: {trainer.local_rank}")
    print(f"World Size: {trainer.world_size}")

    dm = DeltaDataModule()
    model = LitCVNet(num_classes=6)
    #trainer.fit(model, dm)
    trainer.fit(model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())
    print("Training done!")

    print("Test done!")
    
    # Uncomment this if you are using DDP. 
    # AutoLog does not yet work well with the DDP. 
    # The best Practice with DDP would be to track your loss/acc with the Logger 
    # and then log your best_checkpoint back to the mlflow. 
    # In our case we use single node training so you will spot that the acc and loss were tracked. 
    if strategy == "ddp":
        if trainer.global_rank == 0:
            # AutoLog does not work with DDP 
            from mlflow.models.signature import infer_signature
            with mlflow.start_run(run_name="running_cv_uc") as run:
                
                # Train the model ⚡🚅⚡
                print("We are logging our model")
                reqs = mlflow.pytorch.get_default_pip_requirements() + [
                    "pytorch-lightning==" + pl.__version__,
                    "deltalake==0.14.0","deltatorch==0.0.3"
                ]

                mlflow.pytorch.log_model(
                    artifact_path="model_cv_uc",
                    pytorch_model=model.model,
                    pip_requirements=reqs,
                )
                mlflow.set_tag("ml2action", "cv_uc")


# COMMAND ----------

train_distributed(MAX_EPOCH_COUNT, STRATEGY)

# COMMAND ----------

# MAGIC %md 
# MAGIC To do the same but scaling vertically and horizontally if necessary use SparkTorchDistributor, just import the library and change the strategy mode for DDP or FSDP(if you are running GenAI models).
# MAGIC
# MAGIC Here is an example where we have run same code with 4 GPU on 1 single node, if you have multi node set, change the amount of nodes you have. 
# MAGIC ```
# MAGIC from pyspark.ml.torch.distributor import TorchDistributor
# MAGIC
# MAGIC distributed = TorchDistributor(num_processes=4, local_mode=True, use_gpu=True)
# MAGIC distributed.run(train_distributed, 1, "ddp")
# MAGIC ```
# MAGIC
# MAGIC Warning: this package works with 1 GPU per process, and it's in general not recommended to mix nthreads when you have more than 1 process.  
# MAGIC

# COMMAND ----------


