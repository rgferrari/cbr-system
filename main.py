import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from cbr import CBR


def evaluate_performance(cbr, test_set):
    y_true = []
    y_pred = []

    for idx, row in test_set.iterrows():
        query = row.to_dict()
        true_label = query.pop("target")
        result = cbr.retrieve(query)
        predicted_case = result.ranking[0]
        predicted_label = result.casebase[predicted_case]["target"]

        y_true.append(true_label)
        y_pred.append(predicted_label)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")


def main():
    iris = load_iris()
    iris_data = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_data["target"] = iris.target

    train_set, test_set = train_test_split(iris_data, test_size=0.2, random_state=42)

    print(train_set.head())

    case_base = {idx: row.to_dict() for idx, row in train_set.iterrows()}

    config = {
        "sepal length (cm)": "numeric",
        "sepal width (cm)": "numeric",
        "petal length (cm)": "numeric",
        "petal width (cm)": "numeric",
    }

    ga_config = {
        "generations": 15,
        "metric": "accuracy",
        "population_config": {
            "population_size": 70,
            "mutation_rate": 0.4,
            "mutate_individually": True,
            "k_elitism": 10,
            "tournament_size": 15,
        },
    }

    cbr = CBR(case_base, config, use_ga_optimizer=True, ga_config=ga_config)

    # query = {
    #     "sepal length (cm)": 5.1,
    #     "sepal width (cm)": 3.5,
    #     "petal length (cm)": 1.4,
    #     "petal width (cm)": 0.2,
    # }

    # result = cbr.retrieve(query)

    # cbr.print(result)

    print("\n")
    evaluate_performance(cbr, test_set)


if __name__ == "__main__":
    main()
