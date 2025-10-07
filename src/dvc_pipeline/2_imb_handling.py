import yaml
import pandas as pd
from pathlib import Path
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler


def load_data(input_path):
    return pd.read_csv(input_path)


def save_data(df, output_path):
    df.to_csv(output_path, index=False)


def over_sample(df, over_sampling_method):
    X = df.drop(columns=["Sentiment"])
    y = df[["Sentiment"]]

    new_X = None
    new_y = None
    if over_sampling_method == "random":
        rnd = RandomOverSampler(random_state=42)
        new_X, new_y = rnd.fit_resample(X, y)
        new_X.reset_index(drop=True, inplace=True)
        new_y.reset_index(drop=True, inplace=True)

    elif over_sampling_method == "SMOTE":
        rnd = SMOTE(random_state=42, k_neighbors=10)
        new_X, new_y = rnd.fit_resample(X, y)
        new_X.reset_index(drop=True, inplace=True)
        new_y.reset_index(drop=True, inplace=True)
    else:
        new_X = X
        new_y = y

    return pd.concat((new_X, new_y), axis=1)


def main():
    curr_path = Path(__file__)
    home_dir = curr_path.parent.parent.parent
    input_path = home_dir / "data" / "train_test_split"
    output_path = home_dir / "data" / "oversampled"
    output_path.mkdir(parents=True, exist_ok=True)

    with open(home_dir / "params.yaml", "r") as f:
        params = yaml.safe_load(f)["imb"]

    train = load_data(input_path / "train.csv")
    test = load_data(input_path / "test.csv")
    train = over_sample(train, params["over_sampling_tech"])

    save_data(train, output_path / "train.csv")
    save_data(test, output_path / "test.csv")


if __name__ == "__main__":
    main()
