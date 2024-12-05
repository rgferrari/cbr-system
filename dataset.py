import pandas as pd
from pprint import pprint
from ucimlrepo import fetch_ucirepo
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def heart_disease(
    validation=True,
) -> (
    tuple[pd.DataFrame, pd.DataFrame, dict, dict]
    | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]
):
    heart_disease = fetch_ucirepo(id=45)

    df = heart_disease.data.features
    df["target"] = heart_disease.data.targets

    df.fillna(-1, inplace=True)

    pca = PCA()
    pca.fit(df.drop("target", axis=1))

    pooling_weights = {
        feature: float(ratio)
        for feature, ratio in zip(
            df.drop("target", axis=1).columns, pca.explained_variance_ratio_
        )
    }

    train_set, test_set = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["target"]
    )

    if validation is True:
        train_set, validation_set = train_test_split(
            train_set, test_size=0.2, random_state=42, stratify=train_set["target"]
        )

    print("Heart Disease Dataset")
    print("=====================\n")

    print("Train Sample:")
    print(train_set.head())

    heart_disease.variables.replace({"Integer": "numeric"}, inplace=True)

    config = {
        row["name"]: row["type"].lower()
        for _, row in heart_disease.variables.iterrows()
    }

    del config["num"]

    print("\nConfig:")
    pprint(config)

    if validation is True:
        return train_set, validation_set, test_set, config, pooling_weights
    else:
        return train_set, test_set, config, pooling_weights


def iris(
    validation=False,
) -> (
    tuple[pd.DataFrame, pd.DataFrame, dict, dict]
    | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]
):
    iris = load_iris()
    iris_data = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_data["target"] = iris.target

    pca = PCA()
    pca.fit(iris_data.drop("target", axis=1))

    pooling_weights = {
        feature: float(ratio)
        for feature, ratio in zip(
            iris_data.drop("target", axis=1).columns, pca.explained_variance_ratio_
        )
    }

    train_set, test_set = train_test_split(
        iris_data, test_size=0.2, random_state=42, stratify=iris_data["target"]
    )

    if validation is True:
        train_set, validation_set = train_test_split(
            train_set, test_size=0.2, random_state=42, stratify=train_set["target"]
        )

    print("Iris Dataset")
    print("============\n")

    print("Train Sample:")
    print(train_set.head())

    config = {
        "sepal length (cm)": "numeric",
        "sepal width (cm)": "numeric",
        "petal length (cm)": "numeric",
        "petal width (cm)": "numeric",
    }

    print("\nConfig:")
    pprint(config)

    if validation is True:
        return train_set, validation_set, test_set, config, pooling_weights
    else:
        return train_set, test_set, config, pooling_weights


if __name__ == "__main__":
    heart_disease()
