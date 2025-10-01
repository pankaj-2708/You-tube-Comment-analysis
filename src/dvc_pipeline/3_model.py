import pandas as pd
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import dagshub
import mlflow
import yaml
from pathlib import Path
dagshub.init(repo_owner='pankaj-2708', repo_name='You-tube-Comment-analysis', mlflow=True)

mlflow.set_experiment("Python")

def load_data(input_path):
    return pd.read_csv(input_path)


def save_data(df, output_path):
    df.to_csv(output_path , index=False)

def try_models(X, y, test_X,test_y,no_of_features,ngram_range,count_ve,over_sampling_tech, add_info=""):
    for name, model in [
        # (f"xgboost {add_info}", XGBClassifier(device='gpu')),
        # (f"random_forest {add_info}", RandomForestClassifier(n_jobs=-1)),
        # (f"gradient_boosting {add_info}", GradientBoostingClassifier()),
        # (f"cbr {add_info}", CatBoostClassifier(devices='gpu')),
        (f"lgm {add_info}", LGBMClassifier(boosting_type="goss",n_jobs=-1))
    ]:
        with mlflow.start_run():
            mlflow.log_param("model", name)
            mlflow.log_param("over_sampling_tech", over_sampling_tech)
            mlflow.log_param("no of features", no_of_features)
            mlflow.log_param("ngram_range", ngram_range)
            mlflow.log_param("count_vectoriser", count_ve)

            model_ = model

            model_.fit(X, y)
            param=model_.get_params()
            model_.fit(X, y)
            for i in range(len(param)):
                mlflow.log_param(list(param.keys())[i],list(param.values())[i])

            pred_y = model_.predict(test_X)
            pred_y_train = model_.predict(X)

            mlflow.log_metric("accuracy_cv_score", cross_val_score(model,X,y).mean())
            mlflow.log_metric("accuracy", accuracy_score(test_y, pred_y))
            mlflow.log_metric("accuracy_train", accuracy_score(y, pred_y_train))
            
def main():
    curr_path = Path(__file__)
    home_dir = curr_path.parent.parent.parent
    input_path = home_dir / "data" / "train_test_split"
    output_path = home_dir / "models"
    output_path.mkdir(parents=True, exist_ok=True)

    with open(home_dir / "params.yaml", "r") as f:
        params = yaml.safe_load(f)["transform"]
        params2 = yaml.safe_load(f)["imb"]
        
    train = load_data(input_path / "train.csv")
    test = load_data(input_path / "test.csv")
    X=train.drop(columns=['lb__Sentiment'])
    y=train['lb__Sentiment']
    test_X=test.drop(columns=['lb__Sentiment'])
    test_y=test['lb__Sentiment']

    try_models(X,y,test_X,test_y,params['no_of_features'],params['ngram_range'],params['count_vec'],params['over_sampling_tech'])
    
if __name__=="__main__":
    main()