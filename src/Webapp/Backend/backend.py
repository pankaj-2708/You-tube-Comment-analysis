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
    
    
def load_deps(model_name,model_version,run_id="ff50171cbd6d47ffaaf9c4e8d9f41fcd"):
    mlflow.set_tracking_uri("http://ec2-13-53-126-63.eu-north-1.compute.amazonaws.com:5000/")
    client=mlflow.tracking.MlflowClient()
    model_uri=f"models:/{model_name}/{model_version}"
    print(client.download_artifacts(run_id,'vectoriser.pkl'))
    vectoriser=joblib.load(client.download_artifacts(run_id,'vectoriser.pkl'))
    clm_trans=joblib.load(client.download_artifacts(run_id,'clm_trans.pkl'))
    model=mlflow.pyfunc.load_model(model_uri)
    return model,vectoriser,clm_trans

model,vectoriser,clm_trans=load_deps('best_model_ann',1)

@app.get("/predict")
def predict_cat(inp:custom_dtp):
    
    print(inp.txt)
    print("i am good")
    df=pd.DataFrame(inp.txt,columns=['Comment'])
    print("i am good")
    df=preprocess_data(df)
    
    bag_of_words=vectoriser.transform(df['Comment'])
    bag_of_words=pd.DataFrame(bag_of_words.toarray(),columns=vectoriser.get_feature_names_out())
    
    df.reset_index(inplace=True,drop=True)
    df=pd.concat([df,bag_of_words],axis=1)
    df.drop(columns="Comment",inplace=True)
    
    df=pd.DataFrame(clm_trans.transform(df),columns=clm_trans.get_feature_names_out())
    
    output=model.predict(df).argmax(axis=1)
    
    # ['negative', 'neutral', 'positive']
    # matched_output=[]
    # if output==0:
    #     matched_output.append('negative')
    # elif output==1:
    #     matched_output.append('neutral')
    # else:
    #     matched_output.append('positive')
        
    return JSONResponse(status_code=200,content={i:output[i] for i in range(len(output))})