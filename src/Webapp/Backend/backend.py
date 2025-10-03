from fastapi import FastAPI
import mlflow
import joblib


def load_deps(model_name,model_version):
    mlflow.set_tracking_uri("http://ec2-13-53-126-63.eu-north-1.compute.amazonaws.com:5000/")
    client=mlflow.tracking.MlflowClient()
    model_uri=f"models:/{model_name}/{model_version}"
    model=mlflow.pyfunc.load_model(model_uri)
    vectoriser=joblib.load('./')