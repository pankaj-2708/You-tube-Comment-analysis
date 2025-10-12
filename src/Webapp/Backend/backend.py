from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from backend_utility import preprocess_data, get_comments ,genrate_wordcloud ,generate_pie_chart ,generate_trend_chart
from fastapi.responses import JSONResponse
import mlflow
import pandas as pd
import time
import asyncio
import joblib
from pydantic import BaseModel, Field
from functools import lru_cache

# for docker/linux
import tflite_runtime.interpreter as tflite

# for windows
# import tensorflow as tf
# tflite = tf.lite

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class custom_dtp(BaseModel):
    video_id: str = Field(title="video id", description="enter your video id here")


@lru_cache()
def load_deps(run_id="7aacaf0572f0419196f5095d4f1dc11c"):
    # ann oversampled model hyp tunned run id
    mlflow.set_tracking_uri("http://ec2-13-53-126-63.eu-north-1.compute.amazonaws.com:5000/")

    client = mlflow.tracking.MlflowClient()
    # model_uri = f"models:/{model_name}/{model_version}"
    # model = mlflow.pyfunc.load_model(model_uri)
    
    runs = mlflow.search_runs(experiment_names=["HYP tunning"])
    
    vectoriser = joblib.load(client.download_artifacts(run_id, "vectoriser.pkl"))
    clm_trans_path=client.download_artifacts(run_id, "clm_trans.pkl")
    clm_trans = joblib.load(clm_trans_path)

    model_type=runs[runs['run_id']==run_id]['params.type_of_model'].values[0]
    model=None
    if model_type=='ml':
        model_path = client.download_artifacts(run_id, "model.pkl")
        model = joblib.load(model_path)
    else:
        model_path = client.download_artifacts(run_id, "model.tflite")
        model = tflite.Interpreter(model_path=model_path)

    return model_type,model, vectoriser, clm_trans


model_type,interpreter, vectoriser, clm_trans = load_deps()



@app.get("/predict")
def predict_cat(
    video_id: str = Query(..., title="video id", description="enter your video id here")
):

    start_time = time.time()
    total_comments, comments_and_date = asyncio.run(get_comments(video_id))
    if len(comments_and_date) == 0:
        raise HTTPException(status_code=400, detail="failed to fetch comments")
    
    print("time to fetch comments", time.time() - start_time)
    start_time = time.time()
    comments_id = list(comments_and_date.keys())
    comments = list([i[0] for i in comments_and_date.values()])
    pub_date = list([i[1] for i in comments_and_date.values()])

    df = pd.DataFrame(comments, columns=["Comment"])
    df = preprocess_data(df)
    
    print("time to preprocess comments", time.time() - start_time)
    # generate wordcloud , positive negative and neutral percentage over the year , pie chart of percentage , 4 stats and present them in frontend 
    avg_word_count=df['word_count'].mean()
    avg_pos_word_count=df['PositiveWordCount'].mean()
    avg_neg_word_count=df['NegativeWordCount'].mean()
    avg_neu_word_count=df['NeutralWordCount'].mean()
    
    org_df=df.copy()
    

    
    start_time = time.time()
    bag_of_words = vectoriser.transform(df["Comment"])
    bag_of_words = pd.DataFrame(bag_of_words.toarray(), columns=vectoriser.get_feature_names_out())

    df.reset_index(inplace=True, drop=True)
    df = pd.concat([df, bag_of_words], axis=1)
    df.drop(columns="Comment", inplace=True)

    df = pd.DataFrame(clm_trans.transform(df), columns=clm_trans.get_feature_names_out())
    print("time to vectorise comments", time.time() - start_time)
    start_time = time.time()
    output = None
    
    if model_type=='ml':
        output = interpreter.predict(df.values)
    else:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.resize_tensor_input(input_details[0]['index'], [df.shape[0]] + list(input_details[0]['shape'][1:]))
        interpreter.allocate_tensors()
        
        interpreter.set_tensor(input_details[0]['index'], df.values.astype('float32'))
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output = output_data.argmax(axis=1)
    
    # output = model.predict(df.values).argmax(axis=1)
    print("time to predict comments", time.time() - start_time)
    matched_output = []
    for i in output:
        if i == 0:
            matched_output.append("negative")
        elif i == 1:
            matched_output.append("neutral")
        else:
            matched_output.append("positive")

    # print(len(matched_output), len(output), len(comments_id))
    # print(output)
    
    if len(comments_id) != len(output):
        print("length mistmatch", len(comments_id) , len(output))
        
    # print(output)
    # print(org_df.head())
    # print(df.head())
    org_df['Sentiment']=matched_output
    # print(org_df.head())
    wordcloud_neg,wordcloud_neu,wordcloud_pos=genrate_wordcloud(org_df)
    
    
    df=pd.DataFrame({'date':pub_date,'output':output})
    trend_ch=generate_trend_chart(df)
    
    # df.to_csv('a.csv',index=False)
    
    pie_ch=generate_pie_chart(output)
    return JSONResponse(
        status_code=200,
        content={
            "total_comments": total_comments,
            "comments": [
                {"text": comments_id[i], "status": str(matched_output[i])}
                for i in range(len(comments_id))
            ],
            "avg_word_count": round(avg_word_count,2),
            "avg_pos_word_count": round(avg_pos_word_count,2),
            "avg_neg_word_count": round(avg_neg_word_count,2),
            "avg_neu_word_count": round(avg_neu_word_count),
            "wordcloud_neg": wordcloud_neg,
            "wordcloud_neu": wordcloud_neu,
            "wordcloud_pos": wordcloud_pos,
            "pie_chart": pie_ch,
            "trend_chart":trend_ch
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
