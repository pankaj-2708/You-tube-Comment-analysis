import pandas as pd
import mlflow
from pathlib import Path
import yaml
    
def load_data(input_path):
    return pd.read_csv(input_path)

def ann(X,y,test_X,test_y):
    from tensorflow import keras
    from keras import Sequential,regularizers
    from keras.callbacks import EarlyStopping
    from keras.optimizers import RMSprop
    from keras.layers import Dense,BatchNormalization,Dropout
    from sklearn.metrics import accuracy_score,recall_score,precision_score,confusion_matrix,f1_score

    # these are best params models is tunned on xgboost
    best_params={'layer1': 331, 'layer2': 200, 'layer3': 179, 'layer4': 119, 'layer5': 33, 'layer6': 50, 'l2': 0.00010455424293804089, 'activation': 'relu', 'batch_size': 165}

    callback=EarlyStopping(
    monitor='loss',
    patience=10,
    verbose=1,
    mode='auto',
    baseline=None,
    restore_best_weights=False
    )
    model=Sequential()
    model.add(Dense(best_params["layer1"],activation=best_params["activation"],kernel_regularizer=regularizers.L2(best_params["l2"]),kernel_initializer='he_normal',input_shape=(2008,)))
    model.add(BatchNormalization())
    model.add(Dense(best_params["layer2"],kernel_regularizer=regularizers.L2(best_params["l2"]),kernel_initializer='he_normal',activation=best_params["activation"]))
    model.add(Dropout(0.4))
    model.add(Dense(best_params["layer3"],kernel_regularizer=regularizers.L2(best_params["l2"]),kernel_initializer='he_normal',activation=best_params["activation"]))
    model.add(Dropout(0.4))
    model.add(Dense(best_params["layer4"],kernel_regularizer=regularizers.L2(best_params["l2"]),kernel_initializer='he_normal',activation=best_params["activation"]))
    model.add(Dropout(0.4))
    model.add(Dense(best_params["layer5"],kernel_regularizer=regularizers.L2(best_params["l2"]),kernel_initializer='he_normal',activation=best_params["activation"]))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(best_params["layer6"],kernel_regularizer=regularizers.L2(best_params["l2"]),kernel_initializer='he_normal',activation=best_params["activation"]))
    model.add(Dense(3,activation='softmax'))
    model.compile(optimizer='Adam',metrics=["accuracy"],loss="sparse_categorical_crossentropy")
    
    with mlflow.start_run():
        model.fit(X,y,epochs=10,batch_size=best_params["batch_size"],callbacks=callback)
        y_pred=model.predict(test_X).argmax(axis=1)
        pred_y_train=model.predict(X).argmax(axis=1)
        mlflow.log_params(best_params)
        mlflow.log_param("name","ann")
        mlflow.log_metric("test_accuracy",accuracy_score(test_y,y_pred))
        mlflow.log_metric("precison",precision_score(test_y,y_pred))
        mlflow.log_metric("recall_score",recall_score(test_y,y_pred))
        mlflow.log_metric("train_accuracy",accuracy_score(pred_y_train,y))
        mlflow.keras.log_model(model)
        
def main():
    curr_path = Path(__file__)
    home_dir = curr_path.parent.parent.parent
    input_path = home_dir / "data" / "train_test_split"
    output_path = home_dir / "models"
    output_path.mkdir(parents=True, exist_ok=True)

    with open(home_dir / "params.yaml", "r") as f:
        params = yaml.safe_load(f)["dl"]
    
    if not params['model']:
        return 
    mlflow.set_experiment("Deep learning")
    train = load_data(input_path / "train.csv")
    test = load_data(input_path / "test.csv")
    global X,y
    X=train.drop(columns=['lb__Sentiment'])
    y=train['lb__Sentiment']
    test_X=test.drop(columns=['lb__Sentiment'])
    test_y=test['lb__Sentiment']
    ann(X,y,test_X,test_y)
    
if __name__=="__main__":
    main()
    