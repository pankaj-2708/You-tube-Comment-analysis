from fastapi import FastAPI,HTTPException,Query
from fastapi.middleware.cors import CORSMiddleware
from tensorflow import keras
from src.Webapp.Backend.backend_utility import preprocess_data,get_comments
from fastapi.responses import JSONResponse
import mlflow
import pandas as pd
import joblib
from pydantic import BaseModel,Field
from typing import List
from functools import lru_cache

app=FastAPI()
origins = [
    "chrome-extension://egdablaelkefeihkgghmcomnpkibepmn",
    "http://localhost",
    "http://127.0.0.1:5500",  # add your frontend/origin here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class custom_dtp(BaseModel):
    video_id:str=Field(title='video id',description='enter your video id here')
    
@lru_cache()
def load_deps(model_name,model_version,run_id="d03a0797ed8b43d68718f77f7bd6ec83"):
    mlflow.set_tracking_uri("http://ec2-13-53-126-63.eu-north-1.compute.amazonaws.com:5000/")
    
    client=mlflow.tracking.MlflowClient()
    model_uri=f"models:/{model_name}/{model_version}"
    vectoriser=joblib.load(client.download_artifacts(run_id,'vectoriser.pkl'))
    clm_trans=joblib.load(client.download_artifacts(run_id,'clm_trans.pkl'))
    model=mlflow.pyfunc.load_model(model_uri)
    return model,vectoriser,clm_trans

model,vectoriser,clm_trans=load_deps('best_model_ann',1)

import time

import asyncio
@app.get("/predict")
def predict_cat(video_id:str=Query(...,title='video id',description='enter your video id here')):
    
    start_time=time.time()
    total_comments,comments=asyncio.run(get_comments(video_id))
    if len(comments)==0:
        raise HTTPException(status_code=400,detail='failed to fetch comments')
    print("time to fetch comments",time.time()-start_time)
    start_time=time.time()
    comments_id=list(comments.keys())
    comments=list(comments.values())
    
    df=pd.DataFrame(comments,columns=['Comment'])
    df=preprocess_data(df)
    print("time to preprocess comments",time.time()-start_time)
    start_time=time.time()
    bag_of_words=vectoriser.transform(df['Comment'])
    bag_of_words=pd.DataFrame(bag_of_words.toarray(),columns=vectoriser.get_feature_names_out())
    df.reset_index(inplace=True,drop=True)
    df=pd.concat([df,bag_of_words],axis=1)
    df.drop(columns="Comment",inplace=True)
    
    df=pd.DataFrame(clm_trans.transform(df),columns=clm_trans.get_feature_names_out())
    print("time to vectorise comments",time.time()-start_time)
    start_time=time.time()
    
    output=model.predict(df.values).argmax(axis=1)
    print("time to predict comments",time.time()-start_time)
    matched_output=[]
    for i in output:
        if i==0:
            matched_output.append('negative')
        elif i==1:
            matched_output.append('neutral')
        else:
            matched_output.append('positive')
        
    if len(comments_id)!=len(output):
        print("length mistmatch",len(comments_id)!=len(output))
    return JSONResponse(status_code=200,content={"total_comments":total_comments,"comments":[{"text":comments_id[i],"status":str(matched_output[i])} for i in range(len(comments_id))]})