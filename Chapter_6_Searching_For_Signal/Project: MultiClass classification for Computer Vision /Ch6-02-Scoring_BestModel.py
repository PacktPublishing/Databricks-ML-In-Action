# Databricks notebook source
!pip install pytorch-lightning==2.1.2 deltalake==0.14.0 deltatorch==0.0.3 evalidate==2.0.2 pillow==10.1.0
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../../global-setup $project_name=cv_clf

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

# COMMAND ----------

from mlia_utils.cv_clf_funcs import idx_class
train_df = (spark.read.format("delta").load(train_delta_path))
print(idx_class(train_df))

# COMMAND ----------

import os
import mlflow
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlia_utils.cv_clf_funcs import select_best_model

experiment_path = f"/Users/{current_user}/intel-clf-training_action"
local_path = select_best_model(experiment_path)

requirements_path = os.path.join(local_path, "requirements.txt")
if not os.path.exists(requirements_path):
  dbutils.fs.put("file:" + requirements_path, "", True)

device = "cuda" if torch.cuda.is_available() else "cpu"
loaded_model = torch.load(local_path+"/data/model.pth", map_location=torch.device(device))
loaded_model

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Score your model on accuracy 
# MAGIC
# MAGIC **Note** this is a demo purpose model, we have not tried to make the accuracy go higher than what it was.  

# COMMAND ----------

import matplotlib.pyplot as plt
from torch.autograd import Variable
from mlia_utils import cv_clf_funcs as cv_funcs

transform_tests = cv_funcs.transform_imgs()

def pred_class(img):
    # transform images
    img_tens = transform_tests(img)
    # change image format (3,150,150) to (1,3,150,150) by help of unsqueeze function
    # image needs to be in cuda before predition
    img_im = img_tens.unsqueeze(0).cuda() 
    uinput = Variable(img_im)
    uinput = uinput.to(device)
    out = loaded_model(uinput)
    # convert image to numpy format in cpu and snatching max prediction score class index
    index = out.data.cpu().numpy().argmax()    
    return index

# COMMAND ----------

import matplotlib.pyplot as plt
from torch.autograd import Variable
pred_class_func = cv_funcs.pred_class(img)

classes = {k:v for k , v in enumerate(sorted(outcomes))}
loaded_model.eval()

plt.figure(figsize=(20,20))
for i, images in enumerate(pred_files[:10]):
    # just want 25 images to print
    if i > 24:break
    img = Image.open(images)
    index = pred_class_func(img)
    plt.subplot(5,5,i+1)
    plt.title(classes[index])
    plt.axis('off')
    plt.imshow(img)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC **Note** you could also measure your accuracy using directly Lightning or Torch API. 
# MAGIC Here is an example how you could calculate your score on the exact label prediction:
# MAGIC
# MAGIC ```
# MAGIC
# MAGIC test_data = torchvision.datasets.ImageFolder(root=valid_dir, transform=transform_tests)
# MAGIC test_loader= DataLoader(test_data, batch_size=32, shuffle=False, num_workers=2)
# MAGIC
# MAGIC correct_count, all_count = 0,0
# MAGIC pred_label_list = []
# MAGIC proba_list = []
# MAGIC for images, labels in test_loader:
# MAGIC     for i in range(len(labels)):
# MAGIC         if torch.cuda.is_available():
# MAGIC             images = images.cuda()
# MAGIC             labels = labels.cuda()
# MAGIC         
# MAGIC         img = images[i].view(1,3,150,150)
# MAGIC         with torch.no_grad():
# MAGIC             logps = loaded_model(img)
# MAGIC             
# MAGIC         ps = torch.exp(logps)
# MAGIC         probab = list(ps.cpu()[0])
# MAGIC         proba_list.append(probab)
# MAGIC         pred_label = probab.index(max(probab))
# MAGIC         pred_label_list.append(pred_label)
# MAGIC         true_label = labels.cpu()[i]
# MAGIC         if(true_label == pred_label):
# MAGIC             correct_count += 1
# MAGIC         all_count += 1
# MAGIC         
# MAGIC print("Number of images Tested=", all_count)
# MAGIC print("\n Model Accuracy in % =",(correct_count/all_count)*100)
# MAGIC ```

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Score your model to all your images 

# COMMAND ----------

import pandas as pd
from PIL import Image
from torchvision import transforms
import numpy as np
import io
from pyspark.sql.functions import pandas_udf
from typing import Iterator

def feature_extractor(img):
    image = Image.open(io.BytesIO(img))
    transform = transform_tests = torchvision.transforms.Compose([
        transforms.Resize((150,150)),
        transforms.RandomHorizontalFlip(p=0.8), # randomly flip and rotate
        transforms.ColorJitter(0.3,0.4,0.4,0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.425, 0.415, 0.405), (0.205, 0.205, 0.205))
        ])
    return transform(image)

# to reduce time on loading we broadcast our model to each executor 
model_b = sc.broadcast(loaded_model.model)

@pandas_udf("struct< label: int, labelName: string>")
def apply_vit(images_iter: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
    
    model = model_b.value.to(torch.device("cuda"))
    model.eval()
    
    id2label = {0: 'buildings',
                1: 'forest',
                2: 'glacier',
                3: 'mountain',
                4: 'sea',
                5: 'street'}
    
    with torch.set_grad_enabled(False):
        for images in images_iter:
            pil_images = torch.stack(
                [
                    feature_extractor(b)
                    for b in images
                ]
            )
            pil_images = pil_images.to(torch.device("cuda"))
            outputs = model(pil_images)

            preds = torch.max(outputs, 1)[1].tolist()
            probs = torch.nn.functional.softmax(outputs, dim=-1)[:, 1].tolist()
            
            yield pd.DataFrame(
                [
                    {"label": pred, "labelName":id2label[pred]} for pred in preds
                ]
            )

# COMMAND ----------

# with the Brodcasted model we won a few minutes, but it's because we do not have a big dataset, in a case of a big set this could significantly speed up things. 
# also take into account that some models may use Batch Inference natively - check API of your Framework. 
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 32)
predictions_df = spark.read.format("delta").load(f"/Volumes/{catalog}/{database_name}/files/intel_image_clf/valid_imgs_main.delta").withColumn("prediction", apply_vit("content"))
display(predictions_df)

# COMMAND ----------


