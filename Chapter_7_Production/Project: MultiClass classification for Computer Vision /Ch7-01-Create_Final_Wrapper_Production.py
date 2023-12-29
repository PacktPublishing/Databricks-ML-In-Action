# Databricks notebook source
# MAGIC %pip install pytorch-lightning==2.1.2 deltalake==0.14.0 deltatorch==0.0.3 evalidate==2.0.2 pillow==10.1.0

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

import mlflow
from deltatorch import create_pytorch_dataloader, FieldSpec

# COMMAND ----------

from io import BytesIO

class CVModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        # instantiate model in evaluation mode
        model.to(torch.device("cpu"))
        self.model = model.eval()

    def feature_extractor(self, image):
        transform = torchvision.transforms.Compose([
            transforms.Resize((150,150)),
            transforms.RandomHorizontalFlip(p=p), # randomly flip and rotate
            transforms.ColorJitter(0.3,0.4,0.4,0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.425, 0.415, 0.405), (0.205, 0.205, 0.205))
            ])
        # Pay attention here as we will pass oue images in encoded format 
        # this is the  required format by the Serving as of now (has to be DF, array or JSON)
        # so we require to pass a string format. 
        return transform(Image.open(BytesIO(base64.b64decode(image)))) 

    def predict(self, context, images):
        id2label = {
            0: 'buildings',
            1: 'forest',
            2: 'glacier',
            3: 'mountain',
            4: 'sea',
            5: 'street'
            }
        with torch.set_grad_enabled(False):
     
          # add here check if this is a DataFrame 
          # if this is an image remove iterrows 
          pil_images = torch.stack([self.feature_extractor(row[0]) for _, row in images.iterrows()])
          pil_images = pil_images.to(torch.device("cpu"))
          outputs = self.model(pil_images)
          preds = torch.max(outputs, 1)[1]
          probs = torch.nn.functional.softmax(outputs, dim=-1)[:, 1]
          labels = [id2label[pred] for pred in preds.tolist()]

          return pd.DataFrame( 
                              data=dict(
                                label=preds,
                                labelName=labels
                                )
                            )



# COMMAND ----------

from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository

def load_best_model()
  best_model = mlflow.search_runs(
      filter_string=f'attributes.status = "FINISHED"',
      order_by=["metrics.acc DESC"],
      max_results=10,
  ).iloc[0]
  model_uri = "runs:/{}/model".format(best_model.run_id)

  local_path = mlflow.artifacts.download_artifacts(model_uri)
  device = "cuda" if torch.cuda.is_available() else "cpu"

  requirements_path = os.path.join(local_path, "requirements.txt")
  if not os.path.exists(requirements_path):
    dbutils.fs.put("file:" + requirements_path, "", True)


# COMMAND ----------

username = spark.sql("SELECT current_user()").first()["current_user()"]
experiment_path = f"/Users/{username}/intel-clf-training_action"
mlflow.set_experiment(experiment_path)


loaded_model = torch.load(local_path+"/data/model.pth", map_location=torch.device(device))

loaded_model = torch.load(
    local_path + "/data/model.pth", map_location=torch.device("cpu")
)

wrapper = CVModelWrapper(loaded_model)


# COMMAND ----------



# COMMAND ----------

import base64
import pandas as pd

images = spark.read.format("delta").load("/Volumes/ap/cv_uc/intel_image_clf/valid_imgs_main.delta").take(25)

b64image1 = base64.b64encode(images[0]["content"]).decode("ascii")
b64image2 = base64.b64encode(images[1]["content"]).decode("ascii")
b64image3 = base64.b64encode(images[3]["content"]).decode("ascii")
b64image4 = base64.b64encode(images[4]["content"]).decode("ascii")
b64image24 = base64.b64encode(images[24]["content"]).decode("ascii")

df_input = pd.DataFrame(
    [b64image1, b64image2, b64image3, b64image4, b64image24], columns=["data"])

df = wrapper.predict("", df_input)
display(df)

# COMMAND ----------

import mlflow
# Set the registry URI to "databricks-uc" to configure
# the MLflow client to access models in UC
mlflow.set_registry_uri("databricks-uc")
model_name = "ap.cv_uc.cvops_model_mlaction"

from mlflow.models.signature import infer_signature,set_signature
img = df_input['data']
predict_sample = df[['label', 'labelName']]
# To register models under UC you require to log signature for both 
# input and output 
signature = infer_signature(img, predict_sample)

print(f"Your signature is: \n {signature}")

with mlflow.start_run(run_name=model_name) as run:
    mlflowModel = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=wrapper,
        input_example=df_input,
        signature=signature,
        registered_model_name=model_name,
    )
##Alternatively log your model and register later 
# mlflow.register_model(model_uri, "ap.cv_ops.cvops_model")


# COMMAND ----------

from mlflow import MlflowClient
client = MlflowClient()
client.set_registered_model_alias(model_name, "Champion", 1)

model_version_uri = f"models:/{model_name}@Champion"
# Or another option: model_version_uri = f"models:/{model_name}/1"
loaded_model_uc = mlflow.pyfunc.load_model(model_version_uri) # runiid / model_registry_name+version
# champion_version = client.get_model_version_by_alias("prod.ml_team.iris_model", "Champion")

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd
import json

serving_endpoint_name = f"mlaction_endpoint_cv_uc"[:63]

host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
#db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get() 
db_token= dbutils.secrets.get("mlaction", "cv_sp_token")
os.environ['DATABRICKS_TOKEN'] = db_token 
os.environ["DATABRICKS_HOST"] = host


def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
  url = 'https://adb-984752964297111.11.azuredatabricks.net/serving-endpoints/mlaction_endpoint_cv_uc/invocations'
  headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 'Content-Type': 'application/json'}
  ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  data_json = json.dumps(ds_dict, allow_nan=True)
  response = requests.request(method='POST', headers=headers, url=url, data=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()


score_model(df_input)
