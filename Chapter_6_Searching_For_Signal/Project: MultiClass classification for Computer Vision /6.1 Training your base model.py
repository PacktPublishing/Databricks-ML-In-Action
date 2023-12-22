# Databricks notebook source
!pip install pytorch-lightning==2.1.2 deltalake==0.14.0 deltatorch==0.0.3 evalidate==2.0.2 pillow==10.1.0
dbutils.library.restartPython()

# COMMAND ----------

import os
import numpy as np
import io
import logging
from math import ceil

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import transforms, models
import pytorch_lightning as pl
from torchmetrics import Accuracy

from pyspark.sql.functions import col
from PIL import Image

import mlflow
from deltatorch import create_pytorch_dataloader, FieldSpec

# COMMAND ----------

train_delta_path = "/Volumes/{catalog_name}}/{schema_name}/intel_image_clf/train_imgs_main.delta"
val_delta_path = "/Volumes/{catalog_name}/{schema_name}/intel_image_clf/valid_imgs_main.delta"

train_df = (spark.read.format("delta")
            .load(train_delta_path)
            ).limit(10)

display(train_df)

# COMMAND ----------

import mlflow

username = spark.sql("SELECT current_user()").first()["current_user()"]
experiment_path = f"/Users/{username}/intel-clf-training_action"

# This is needed for later in the notebook
db_host = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .extraContext()
    .apply("api_url")
)
db_token = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
)

# create experiment if does not exist 
def mlflow_set_experiment(experiment_path:str = None):
    try:
        print(f"Setting our existing experiment {experiment_path}")
        mlflow.set_experiment(experiment_path)
        experiment = mlflow.get_experiment_by_name(experiment_path)
    except:
        print("Creating a new experiment and setting it")
        experiment = mlflow.create_experiment(name = experiment_path)
        mlflow.set_experiment(experiment_id=experiment_path)

mlflow_set_experiment(experiment_path) 
log_path = "/Volumes/ap/cv_uc/intel_image_clf/intel_training_logger_board"


# COMMAND ----------

import torchvision
class LitCVNet(pl.LightningModule):
        # we can also define model in this Module or we can define in standard pytorch Module
        # then wrap with Pytorch-Lightning Module , You can save & load model weights without 
        # altering pytorch / Lightning module . You will learn in the later series .
        def __init__(self, num_classes = 6, learning_rate= 0.0001, family = "resnext", momentum = 0.8):
            super().__init__()
            self.family = family
            self.momentum = momentum
            self.accuracy = Accuracy(task="multiclass", num_classes=6)
            self.learning_rate = learning_rate 
            self.model = self.get_model(num_classes)
            self.criterion = nn.CrossEntropyLoss()

        def get_model(self, num_classes):
            """
            This is the function that initialises our model.
            If we wanted to use other prebuilt model libraries like timm we would put that model here
            """
            backbone = torchvision.models.wide_resnet50_2(pretrained=True)
            for param in backbone.parameters():
                param.required_grad = False
            num_ftrt = backbone.fc.in_features
            backbone.fc = nn.Linear(num_ftrt, 6)
            return backbone

        # We do not overwrite our forward pass 
        def forward(self, x):
            x  = self.model(x)
            return x
        
        def training_step(self,batch,batch_idx):
            x = batch["content"]
            y = batch["label_id"]
            outputs = self.forward(x)
            loss = self.criterion(outputs,y)
            acc = self.accuracy(outputs,y)
            self.log("train_loss", torch.tensor([loss]), on_step=True, on_epoch=True, logger=True)
            self.log("train_acc", torch.tensor([acc]), on_step=True, on_epoch=True, logger=True)
            return loss
        
        def validation_step(self,batch,batch_idx):
            x = batch["content"]
            y = batch["label_id"]
            outputs = self.forward(x)
            loss = self.criterion(outputs,y)
            acc = self.accuracy(outputs, y)
            self.log("val_loss", torch.tensor([loss]), on_step=True, logger=True)
            self.log("val_acc", torch.tensor([acc]), prog_bar=True, logger=True)
            return {"loss": loss, "acc": acc}
        
        # predict_step is optional unless you are doing some predictions
        def predict_step(self, batch, batch_idx, dataloader_idx=0):
            x = batch["content"]
            y = batch["label_id"]
            preds = self.model(x)
            return preds
        
        def configure_optimizers(self):
            params = self.model.fc.parameters()
            #optimizer = torch.optim.AdamW(params, lr=self.learning_rate)
            optimizer = torch.optim.SGD(params, lr=self.learning_rate)
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4,6], gamma=0.06)
            # {"optimizer": `Optimizer`, (optional) "lr_scheduler": `LRScheduler`}
            return {"optimizer":optimizer, "lr_scheduler":lr_scheduler}


# COMMAND ----------

class DeltaDataModule(pl.LightningDataModule):
    """
    Creating a Data loading module with Delta Torch loader 
    """
    def __init__(self):
        super().__init__()
        self.num_classes = 6

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
            shuffle=False,
            batch_size=batch_size,
        )

    def train_dataloader(self):
        return self.dataloader(
            train_delta_path,
            batch_size=100,
        )

    def val_dataloader(self):
        return self.dataloader(val_delta_path, batch_size=50)

    def test_dataloader(self):
        return self.dataloader(val_delta_path, batch_size=30)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Native Torch Loader 
# MAGIC If you would like to keep your images under Volumes and load them from there here is an example for Native Loader 
# MAGIC
# MAGIC ```
# MAGIC
# MAGIC class DeltaDataModule(pl.LightningDataModule):
# MAGIC     """
# MAGIC     Creating a Data loading module with Delta Torch loader 
# MAGIC     """
# MAGIC     def __init__(self, train_dir, valid_dir):
# MAGIC         super().__init__()
# MAGIC
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
# MAGIC         self.train_data = torchvision.datasets.ImageFolder(root=train_dir, transform=self.transform)
# MAGIC         self.test_data = torchvision.datasets.ImageFolder(root=valid_dir, transform=self.transform_tests)
# MAGIC         
# MAGIC         self.train_sampler, self.valid_sampler = self.shuffle_data(self.train_data,)
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
# MAGIC
# MAGIC
# MAGIC ```

# COMMAND ----------

#Check GPU availability
if not torch.cuda.is_available(): # is gpu
  raise Exception("Please use a GPU-cluster for model training, CPU instances will be too slow")

# COMMAND ----------

MAX_EPOCH_COUNT = 10
STEPS_PER_EPOCH = 5
EARLY_STOP_MIN_DELTA = 0.1
EARLY_STOP_PATIENCE = 10

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
        monitor="train_loss", # this has been saved under the Model Trainer - inside the validation_step function 
        dirpath=log_path,
        filename="sample-cvops-{epoch:02d}-{val_loss:.2f}"
    )
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="train_acc",
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
           checkpoint_callback,
           tqdm_callback
        ],
    )

    print(f"Global Rank: {trainer.global_rank}")
    print(f"Local Rank: {trainer.local_rank}")
    print(f"World Size: {trainer.world_size}")

    dm = DeltaDataModule()
    model = LitCVNet(num_classes=6)
    trainer.fit(model, dm)
    print("Training done!")

    print("Test done!")

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

train_distributed(10, "auto")

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
# MAGIC Warning: this package works with 1 GPU per process. 
# MAGIC
