import yaml
import pandas as pd
from pathlib import Path
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

def load_data(input_path):
    return pd.read_csv(input_path)

def save_data(df, output_path):
    df.to_csv(output_path , index=False)
    
def over_sample(df,over_sampling_method):
    X=df.drop(columns=['lb__Sentiment'])
    y=df[['lb__Sentiment']]
    
    new_X=None
    new_y=None
    if over_sampling_method=="random":
        rnd=RandomOverSampler(random_state=42)
        new_X,new_y=rnd.fit_resample(X,y)
        new_X.reset_index(drop=True,inplace=True)
        new_y.reset_index(drop=True,inplace=True)
        
    elif over_sampling_method=="SMOTE":
        rnd=SMOTE(random_state=42,k_neighbors=10)
        new_X,new_y=rnd.fit_resample(X,y)
        new_X.reset_index(drop=True,inplace=True)
        new_y.reset_index(drop=True,inplace=True)
    else:
        new_X=X
        new_y=y
        
    return pd.concat((new_X,new_y),axis=1)

def split(df,output_path,test_size):
    train,test=train_test_split(df,test_size=test_size)
    save_data(train,output_path / 'train.csv')
    save_data(test,output_path / 'test.csv')
    
def main():
    curr_path = Path(__file__)
    home_dir = curr_path.parent.parent.parent
    input_path = home_dir / "data" / "transformed"
    output_path = home_dir / "data" / "train_test_split"
    output_path.mkdir(parents=True, exist_ok=True)

    with open(home_dir / "params.yaml", "r") as f:
        params = yaml.safe_load(f)["imb"]
        
    df = load_data(input_path / "transformed.csv")

    split(over_sample(df,params["over_sampling_tech"]),output_path,params['test_size'])
    
if __name__=="__main__":
    main()