from fastapi import FastAPI,HTTPException
from tensorflow import keras
from src.Webapp.Backend.backend_utility import preprocess_data
from fastapi.responses import JSONResponse
import mlflow
import pandas as pd
import joblib
from pydantic import BaseModel
from typing import List

app=FastAPI()

class custom_dtp(BaseModel):
    txt:List[str]
    
    
def load_deps(model_name,model_version,run_id="d03a0797ed8b43d68718f77f7bd6ec83"):
    mlflow.set_tracking_uri("http://ec2-13-53-126-63.eu-north-1.compute.amazonaws.com:5000/")
    client=mlflow.tracking.MlflowClient()
    model_uri=f"models:/{model_name}/{model_version}"
    vectoriser=joblib.load(client.download_artifacts(run_id,'vectoriser.pkl'))
    clm_trans=joblib.load(client.download_artifacts(run_id,'clm_trans.pkl'))
    model=mlflow.pyfunc.load_model(model_uri)
    return model,vectoriser,clm_trans

model,vectoriser,clm_trans=load_deps('best_model_ann',1)

@app.get("/predict")
def predict_cat(inp:custom_dtp):
    
    df=pd.DataFrame(inp.txt,columns=['Comment'])
    df=preprocess_data(df)
    
    bag_of_words=vectoriser.transform(df['Comment'])
    bag_of_words=pd.DataFrame(bag_of_words.toarray(),columns=vectoriser.get_feature_names_out())
    
    df.reset_index(inplace=True,drop=True)
    df=pd.concat([df,bag_of_words],axis=1)
    df.drop(columns="Comment",inplace=True)
    
    df=pd.DataFrame(clm_trans.transform(df),columns=clm_trans.get_feature_names_out())
    
    output=model.predict(df.values).argmax(axis=1)
    
    # ['negative', 'neutral', 'positive']
    # matched_output=[]
    # if output==0:
    #     matched_output.append('negative')
    # elif output==1:
    #     matched_output.append('neutral')
    # else:
    #     matched_output.append('positive')
        
    return JSONResponse(status_code=200,content={str(i):str(output[i]) for i in range(len(output))})