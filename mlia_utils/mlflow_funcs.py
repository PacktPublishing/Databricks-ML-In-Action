### mlflow functions
from mlflow.tracking import MlflowClient
import mlflow

def get_latest_model_version(model_name:str = None):
    """
    Getting your latest version of the registered MLFlow model
    
    """
    latest_version = 1
    mlflow_client = MlflowClient()
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
      version_int = int(mv.version)
      if version_int > latest_version:
        latest_version = version_int
    return latest_version
 
def mlflow_set_experiment(experiment_path:str = None):
    """
    Create or set your MLFlow experiment 
    experiment_path:str :: path where you want your experiment to be stored 
    """
    try:
        print(f"Setting our existing experiment {experiment_path}")
        mlflow.set_experiment(experiment_path)
        experiment = mlflow.get_experiment_by_name(experiment_path)
    except:
        print("Creating a new experiment and setting it")
        experiment = mlflow.create_experiment(name = experiment_path)
        mlflow.set_experiment(experiment_id=experiment_path)