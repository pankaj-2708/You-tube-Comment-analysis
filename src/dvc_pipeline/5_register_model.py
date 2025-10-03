import pandas as pd
import mlflow
from pathlib import Path
import yaml

mlflow.set_tracking_uri("http://ec2-13-53-126-63.eu-north-1.compute.amazonaws.com:5000/")

def register_best_model():
    runs = mlflow.search_runs(experiment_names=['HYP tunning'])
    # print(runs.columns)
    runs.sort_values(by="metrics.accuracy", ascending=False, inplace=True)
    best_run = runs.iloc[0]
    best_run_id = best_run["run_id"]

    # print("best_model_details")
    # print(best_run)
    # print(best_run["artifact_uri"])

    model_name = f"best_model_{best_run['params.model']}"
    run_id = best_run_id
    
    # model in model uri come from the artifact path given while logging the model
    model_uri = f"runs:/{run_id}/model"
    model_version=mlflow.register_model(model_uri,model_name)

    # changeing model stage to stagging
    # client=mlflow.tracking.MlflowClient()
    # client.transition_model_version_stage(
    #     name=model_name,
    #     version=model_version.version,
    #     stage='Staging'
    # )
    
def main():
    curr_path = Path(__file__)
    home_dir = curr_path.parent.parent.parent
    
    with open(home_dir / "params.yaml", "r") as f:
        params = yaml.safe_load(f)["register_model"]
        
    if params['register']:
        register_best_model()
    
if __name__=="__main__":
    register_best_model()