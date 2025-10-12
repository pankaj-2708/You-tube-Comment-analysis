import pandas as pd
from pathlib import Path
import mlflow
import yaml
import optuna

mlflow.set_tracking_uri("http://ec2-13-53-126-63.eu-north-1.compute.amazonaws.com:5000/")


X, y = None, None


def load_data(input_path):
    return pd.read_csv(input_path)


def save_data(df, output_path):
    df.to_csv(output_path, index=False)


def try_models(
    X, y, test_X, test_y, no_of_features, ngram_range, count_ve, over_sampling_tech, add_info=""
):
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from lightgbm import LGBMClassifier
    from sklearn.metrics import accuracy_score
    from xgboost import XGBClassifier
    from sklearn.model_selection import cross_val_score

    for name, model in [
        # (f"xgboost {add_info}", XGBClassifier()),
        (f"random_forest {add_info}", RandomForestClassifier(n_jobs=-1, random_state=42)),
        # (f"gradient_boosting {add_info}", GradientBoostingClassifier(random_state=42)),
        # (f"lgm {add_info}", LGBMClassifier(boosting_type="goss", n_jobs=-1, random_state=42)),
        # (f"svc {add_info}", SVC(random_state=42)),
    ]:
        with mlflow.start_run():
            mlflow.log_param("type_of_model", "ml")
            mlflow.log_param("model", name)
            mlflow.log_param("over_sampling_tech", over_sampling_tech)
            mlflow.log_param("no of features", no_of_features)
            mlflow.log_param("ngram_range", ngram_range)
            mlflow.log_param("count_vectoriser", count_ve)

            model_ = model

            model_.fit(X, y)
            param = model_.get_params()
            # model_.fit(X, y)
            for i in range(len(param)):
                mlflow.log_param(list(param.keys())[i], list(param.values())[i])

            pred_y = model_.predict(test_X)
            pred_y_train = model_.predict(X)

            mlflow.log_metric("accuracy_cv_score", cross_val_score(model, X, y).mean())
            mlflow.log_metric("accuracy", accuracy_score(test_y, pred_y))
            mlflow.log_metric("accuracy_train", accuracy_score(y, pred_y_train))
            # mlflow.pyfunc.log_model(model_,artifact_path='model')


def objective_XGB(trial):
    from sklearn.model_selection import cross_val_score
    from xgboost import XGBClassifier

    global X, y
    n_estimators_ = trial.suggest_int("n_estimators", 10, 200)
    max_depth_ = trial.suggest_int("max_depth", 3, 40)
    eta_ = trial.suggest_float("eta", 0.01, 1, log=True)
    sampling_method_ = trial.suggest_categorical("sampling_method", ["uniform", "gradient_based"])
    tree_method_ = trial.suggest_categorical("tree_method", ["hist"])
    subsample_ = trial.suggest_float("subsample", 0.1, 1)
    colsample_bytree_ = trial.suggest_float("colsample_bytree", 0.1, 1)
    colsample_bylevel_ = trial.suggest_float("colsample_bylevel", 0.1, 1)
    colsample_bynode_ = trial.suggest_float("colsample_bynode", 0.1, 1)
    reg_lambda_ = trial.suggest_float("reg_lambda", 0, 10)
    reg_alpha_ = trial.suggest_float("reg_alpha", 0, 5)
    gamma_ = trial.suggest_float("gamma", 0, 10)

    model = XGBClassifier(
        n_estimators=n_estimators_,
        max_depth=max_depth_,
        learning_rate=eta_,
        sampling_method=sampling_method_,
        tree_method=tree_method_,
        subsample=subsample_,
        colsample_bylevel=colsample_bylevel_,
        colsample_bynode=colsample_bynode_,
        colsample_bytree=colsample_bytree_,
        reg_alpha=reg_alpha_,
        reg_lambda=reg_lambda_,
        gamma=gamma_,
    )

    score = cross_val_score(model, X=X, y=y, cv=5).mean()

    return score


def objective_RF(trial):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    global X, y
    n_estimators_ = trial.suggest_int("n_estimators", 1, 2)
    max_depth_ = trial.suggest_int("max_depth", 1, 2)
    max_features_ = trial.suggest_float("max_features", 0, 1)
    bootstrap_ = trial.suggest_categorical("bootstrap", [True, False])
    crietion_ = trial.suggest_categorical("criterion", ["gini", "entropy"])
    model = None
    if bootstrap_:
        max_samples_ = trial.suggest_float("max_samples", 0, 1)
        model = RandomForestClassifier(
            n_estimators=n_estimators_,
            max_depth=max_depth_,
            max_samples=max_samples_,
            max_features=max_features_,
            bootstrap=bootstrap_,
            criterion=crietion_,
            random_state=42,
        )
    else:
        model = RandomForestClassifier(
            n_estimators=n_estimators_,
            max_depth=max_depth_,
            max_features=max_features_,
            bootstrap=bootstrap_,
            criterion=crietion_,
            random_state=42,
        )

    score = cross_val_score(model, X=X, y=y, cv=5).mean()

    return score


def tune_xgb(X, y, test_X, test_Y, home_dir,output_path):
    import plotly.express as px
    from optuna.visualization import plot_optimization_history, plot_slice, plot_param_importances
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.model_selection import cross_val_score
    from xgboost import XGBClassifier

    # include kaleido in deps required for px image export

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective_XGB, n_trials=30)

    best_trial = study.best_trial

    with mlflow.start_run():
        mlflow.log_param("model", "xgboost_tunned")
        mlflow.log_param("type_of_model", "ml")
        for trial in study.get_trials():
            with mlflow.start_run(nested=True):
                j = {k: str(v) for k, v in trial.params.items()}
                mlflow.log_params(j)
                mlflow.log_metric(key="accuracy", value=float(trial.value))

        for key, value in best_trial.params.items():
            mlflow.log_param(key, value)

        model_ = XGBClassifier(**best_trial.params, random_state=42)
        with open(output_path / "model.pkl", "wb") as f:
            import joblib

            joblib.dump(model_, f)
        # also logging model in pickle format for easy loading
        mlflow.log_artifact(output_path / "model.pkl")
        model_.fit(X, y)
        signature = mlflow.models.infer_signature(X, y)
        mlflow.xgboost.log_model(model_, artifact_path="model")

        pred_y = model_.predict(test_X)
        pred_y_train = model_.predict(X)
        fig = px.imshow(confusion_matrix(test_Y, pred_y), text_auto=True)
        mlflow.log_figure(fig, "confusion_mat.png")
        mlflow.log_artifact(home_dir / "data" / "transformed" / "vectoriser.pkl")
        mlflow.log_artifact(home_dir / "data" / "transformed" / "clm_trans.pkl")
        mlflow.log_metric("accuracy", accuracy_score(test_Y, pred_y))
        mlflow.log_metric("accuracy_train", accuracy_score(pred_y_train, y))
        fig = plot_optimization_history(study)
        mlflow.log_figure(fig, "optuna_optimization_history.png")
        fig = plot_param_importances(study)
        mlflow.log_figure(fig, "optuna_param_importance.png")
        fig = plot_slice(study)
        mlflow.log_figure(fig, "optuna_plot_slice.png")


def tune_rf(X, y, test_X, test_Y, home_dir,output_path):
    try:
        import plotly.express as px
        from optuna.visualization import plot_optimization_history, plot_slice, plot_param_importances
        from sklearn.metrics import accuracy_score, confusion_matrix
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())

        study.optimize(objective_RF, n_trials=2)

        best_trial = study.best_trial

        with mlflow.start_run():
            mlflow.log_param("type_of_model", "ml")
            mlflow.log_param("model", "random_forest_tunned")
            for trial in study.get_trials():
                with mlflow.start_run(nested=True):
                    j = {k: str(v) for k, v in trial.params.items()}
                    mlflow.log_params(j)
                    mlflow.log_metric(key="accuracy", value=float(trial.value))

            for key, value in best_trial.params.items():
                mlflow.log_param(key, value)

            model_ = RandomForestClassifier(**best_trial.params, random_state=42)

            model_.fit(X, y)
            mlflow.sklearn.log_model(model_, artifact_path="model")
            with open(output_path / "model.pkl", "wb") as f:
                import joblib

                joblib.dump(model_, f)
            # also logging model in pickle format for easy loading
            mlflow.log_artifact(output_path / "model.pkl")
            pred_y = model_.predict(test_X)
            pred_y_train = model_.predict(X)
            confusion_matrix(test_Y, pred_y)

            mlflow.log_artifact(home_dir / "data" / "train_test_split" / "vectoriser.pkl")
            mlflow.log_artifact(home_dir / "data" / "train_test_split" / "clm_trans.pkl")
            mlflow.log_metric("accuracy", accuracy_score(test_Y, pred_y))
            mlflow.log_metric("accuracy_train", accuracy_score(pred_y_train, y))

            fig = px.imshow(confusion_matrix(test_Y, pred_y), text_auto=True)
            mlflow.log_figure(fig, "confusion_mat.png")
            fig = plot_optimization_history(study)
            mlflow.log_figure(fig, "optuna_optimization_history.png")
            fig = plot_param_importances(study)
            mlflow.log_figure(fig, "optuna_param_importance.png")
            fig = plot_slice(study)
            mlflow.log_figure(fig, "optuna_plot_slice.png")
    except Exception as e:
        print(e)
        print("error in tuning rf")

def main():
    curr_path = Path(__file__)
    home_dir = curr_path.parent.parent.parent
    input_path = home_dir / "data" / "oversampled"
    output_path = home_dir / "models"

    with open(home_dir / "params.yaml", "r") as f:
        params = yaml.safe_load(f)["transform"]
    with open(home_dir / "params.yaml", "r") as f:
        params2 = yaml.safe_load(f)["imb"]

    with open(home_dir / "params.yaml", "r") as f:
        params3 = yaml.safe_load(f)["model"]

    train = load_data(input_path / "train.csv")
    test = load_data(input_path / "test.csv")
    global X, y
    X = train.drop(columns=["Sentiment"])
    y = train["Sentiment"]
    test_X = test.drop(columns=["Sentiment"])
    test_y = test["Sentiment"]

    if params3["tune_xgb"]:
        mlflow.set_experiment("HYP tunning")
        tune_xgb(X, y, test_X, test_y, home_dir,output_path)
    if params3["tune_rf"]:
        mlflow.set_experiment("HYP tunning")
        tune_rf(X, y, test_X, test_y, home_dir,output_path)
    if params3["try_models"]:
        mlflow.set_experiment("Python")
        try_models(
            X,
            y,
            test_X,
            test_y,
            params["no_of_features"],
            params["ngram_range"],
            params["count_vec"],
            params2["over_sampling_tech"],
        )


if __name__ == "__main__":
    main()
