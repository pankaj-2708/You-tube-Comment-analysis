import pandas as pd
from pathlib import Path
import pickle
import yaml
from sklearn.preprocessing import StandardScaler,MinMaxScaler,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer


def load_data(input_path):
    return pd.read_csv(input_path)


def save_data(df, output_path):
    df.to_csv(output_path , index=False)
    

def vectorisation(df,no_of_features,ngram_range,output_path,count_ve):
    df.dropna(inplace=True)
    
    count_vec=None
    if count_ve:
        count_vec=CountVectorizer(max_features=no_of_features,ngram_range=(int(ngram_range.split(",")[0]),int(ngram_range.split(",")[1])))
    else:
        count_vec=TfidfVectorizer(max_features=no_of_features,
                                  ngram_range=(int(ngram_range.split(",")[0]),int(ngram_range.split(",")[1])))
        
    bag_of_words=count_vec.fit_transform(df['Comment'])
    bag_of_words=pd.DataFrame(bag_of_words.toarray(),columns=count_vec.get_feature_names_out())
    df.reset_index(inplace=True,drop=True)
    df=pd.concat([df,bag_of_words],axis=1)
    
    # saving count vectoriser
    with open(output_path / "vectoriser.pkl",'wb') as f:
        pickle.dump(count_vec,f)
        
    df.drop(columns="Comment",inplace=True)
    
    return df

def transform(df,standardise,output_path):
    std=None
    if standardise:
        std=StandardScaler()
    else:
        std=MinMaxScaler()
        
    lb=OrdinalEncoder()

    clm=ColumnTransformer([
        ('std',std,['comment_len','word_count','char_per_words','apos_count','stopword_count','PositiveWordCount','NegativeWordCount','NeutralWordCount']),
        ('lb',lb,['Sentiment'])
    ],
    remainder='passthrough')

    with open(output_path / "clm_trans.pkl",'wb') as f:
        pickle.dump(clm,f)
        
    df=pd.DataFrame(clm.fit_transform(df),columns=clm.get_feature_names_out())
    return df



def main():
    curr_path = Path(__file__)
    home_dir = curr_path.parent.parent.parent
    input_path = home_dir / "data" / "processed" / "preprocessed.csv"
    output_path = home_dir / "data" / "transformed"
    output_path.mkdir(parents=True, exist_ok=True)

    with open(home_dir / "params.yaml", "r") as f:
        params = yaml.safe_load(f)["transform"]
        
    df = load_data(input_path)
    
    df=vectorisation(df,params['no_of_features'],params['ngram_range'],output_path,params['count_vec'])
    df=transform(df,params['standardise'],output_path)
    save_data(df,output_path/"transformed.csv")
    
if __name__=="__main__":
    main()