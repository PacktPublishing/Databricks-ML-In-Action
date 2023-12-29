import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
from PIL import Image
import mlflow 
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
import os 


def transform_imgs(p=0.5):
  return torchvision.transforms.Compose([
      transforms.Resize((150,150)),
      transforms.RandomHorizontalFlip(p=p), # randomly flip and rotate
      transforms.ColorJitter(0.3,0.4,0.4,0.2),
      transforms.ToTensor(),
      transforms.Normalize((0.425, 0.415, 0.405), (0.205, 0.205, 0.205))
      ])

def idx_class(df):            
    unique_object_ids = df.select("label_name").distinct().collect()
    object_id_to_class_mapping = {
        unique_object_ids[idx].label_name: idx for idx in range(len(unique_object_ids))}
    return object_id_to_class_mapping

def select_best_model(experiment_path, artiffact_name = "model"):
  mlflow.set_experiment(experiment_path)
  best_model = mlflow.search_runs(
                filter_string=f'attributes.status = "FINISHED"',
                order_by=["metrics.acc DESC"],
                max_results=10,
                ).iloc[0]
  model_uri = "runs:/{0}/{1}".format(best_model.run_id, artiffact_name)
  local_path = mlflow.artifacts.download_artifacts(model_uri)
  return local_path 

def display_image(path, dpi=50):
    img = Image.open(path)
    width, height = img.size
    plt.figure(figsize=(width / dpi, height / dpi))
    plt.imshow(img, interpolation="nearest", aspect="auto")


def proportion_labels(labels_dict_name):  
  import numpy as np
  final_s = np.zeros(6)
  ticks = []
  idx_array = np.zeros(6)
  
  # plot the pie chart and bar graph of labels
  for idx,(ikey, ival) in enumerate(labels_dict_name.items()):
    print(ival,ikey, idx)
    final_s[idx] = ival
    idx_array[idx] = idx+1
    ticks.append(ikey)
    
  import matplotlib.pyplot as plt
  plt.figure(figsize=(20,9))

  plt.subplot(121)
  plt.bar(idx_array, final_s)
  plt.xticks(ticks=idx_array, labels=ticks, fontsize=12, fontweight='bold')
  plt.yticks(fontsize=12, fontweight='bold')
  plt.grid(visible=True)
  plt.title("Number of images per class", size=14, weight='bold')

  plt.subplot(122)
  plt.pie(final_s.ravel(),
          labels=ticks,
          autopct='%1.2f%%',
          textprops={'fontweight':'bold'}
          )
  plt.title("proportion of classes", size=14, weight='bold')

  plt.suptitle(f"Proportion of data", size=20, weight='bold')
  plt.show()