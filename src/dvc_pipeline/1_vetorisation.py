import pandas as pd
from pathlib import Path
import pickle
import yaml
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def load_data(input_path):
    return pd.read_csv(input_path)


def save_data(df, output_path):
    df.to_csv(output_path, index=False)


def split(df, test_size):
    train, test = train_test_split(
        df, test_size=test_size, stratify=df["Sentiment"], random_state=42
    )
    return train, test

def vectorisation(train, test, no_of_features, ngram_range, output_path, count_ve):
    train.dropna(inplace=True)
    test.dropna(inplace=True)
    count_vec = None
    if count_ve:
        count_vec = CountVectorizer(
            max_features=no_of_features,
            ngram_range=(int(ngram_range.split(",")[0]), int(ngram_range.split(",")[1])),
        )
    else:
        count_vec = TfidfVectorizer(
            max_features=no_of_features,
            ngram_range=(int(ngram_range.split(",")[0]), int(ngram_range.split(",")[1])),
        )

    train.reset_index(inplace=True)
    test.reset_index(inplace=True)
    
    train_y = train["Sentiment"]
    train = train.drop(columns=["Sentiment"])

    test_y = test["Sentiment"]
    test = test.drop(columns=["Sentiment"])

    bag_of_words = count_vec.fit_transform(train["Comment"])
    bag_of_words2 = count_vec.transform(test["Comment"])
    bag_of_words = pd.DataFrame(bag_of_words.toarray(), columns=count_vec.get_feature_names_out())
    bag_of_words2 = pd.DataFrame(
        bag_of_words2.toarray(), columns=count_vec.get_feature_names_out()
    )

    # train.reset_index(inplace=True, drop=True)
    train = pd.concat([train, bag_of_words], axis=1)

    # test.reset_index(inplace=True, drop=True)
    test = pd.concat([test, bag_of_words2], axis=1)

    # saving count vectoriser
    with open(output_path / "vectoriser.pkl", "wb") as f:
        pickle.dump(count_vec, f)

    train.drop(columns="Comment", inplace=True)
    test.drop(columns="Comment", inplace=True)


    train["Sentiment"] = train_y
    test["Sentiment"] = test_y

    return train, test


def transform(train, test, standardise, output_path):
    # print(train.isnull().sum())
    # print(test.isnull().sum())
    # train.dropna(inplace=True)
    # test.dropna(inplace=True)

    std = None
    if standardise:
        std = StandardScaler()
    else:
        std = MinMaxScaler()

    lb = LabelEncoder()

    train_y = train["Sentiment"]
    train = train.drop(columns=["Sentiment"])

    test_y = test["Sentiment"]
    test = test.drop(columns=["Sentiment"])

    clm = ColumnTransformer(
        [
            ("std", std, [i for i in range(train.shape[1])]),
        ],
        remainder="passthrough",
    )

    train = pd.DataFrame(clm.fit_transform(train), columns=clm.get_feature_names_out())

    train_y = lb.fit_transform(train_y)
    print(lb.classes_)

    test = pd.DataFrame(clm.transform(test), columns=clm.get_feature_names_out())
    test_y = lb.transform(test_y)

    with open(output_path / "clm_trans.pkl", "wb") as f:
        pickle.dump(clm, f)

    train["Sentiment"] = train_y
    test["Sentiment"] = test_y

    return train, test


def main():
    curr_path = Path(__file__)
    home_dir = curr_path.parent.parent.parent
    input_path = home_dir / "data" / "processed" / "preprocessed.csv"
    output_path = home_dir / "data" / "train_test_split"
    output_path.mkdir(parents=True, exist_ok=True)

    with open(home_dir / "params.yaml", "r") as f:
        params = yaml.safe_load(f)["transform"]

    df = load_data(input_path)
    print(df.shape)
    train, test = split(df, params["test_size"])

    train, test = vectorisation(
        train,
        test,
        params["no_of_features"],
        params["ngram_range"],
        output_path,
        params["count_vec"],
    )
    # print(train.shape,test.shape)
    train, test = transform(train, test, params["standardise"], output_path)
    # print(train.shape,test.shape)

    save_data(train, output_path / "train.csv")
    save_data(test, output_path / "test.csv")


if __name__ == "__main__":
    main()
