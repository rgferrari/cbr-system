import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from cbr import CBR
from dataset import heart_disease
import json
import numpy as np


def evaluate_performance(cbr, test_set, runs_json=None, run_name=None):
    y_true = []
    y_pred = []

    for idx, row in test_set.iterrows():
        query = row.to_dict()
        true_label = query.pop("target")
        result = cbr.retrieve(query)
        similar_cases = result.ranking[:5]
        labels = [result.casebase[case]["target"] for case in similar_cases]

        values, counts = np.unique(labels, return_counts=True)
        predicted_label = values[np.argmax(counts)]

        y_true.append(true_label)
        y_pred.append(predicted_label)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    print("\nEvaluation:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    if runs_json is not None and run_name is not None:
        runs_json[run_name].update(
            {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

        runs_json[run_name]["pooling_weights"] = cbr.pooling_weights

        with open("runs.json", "w") as f:
            json.dump(runs_json, f)


def main():
    runs_json = json.load(open("runs.json", "r"))
    run_name = ""
    description = ""
    stratified = True
    use_ga_optimizer = True

    runs_json[run_name] = {}

    train_set, validation_set, test_set, config, pooling_weights = heart_disease(
        validation=True, stratify=stratified
    )

    case_base = {idx: row.to_dict() for idx, row in train_set.iterrows()}

    validation_case_base = {
        idx: row.to_dict() for idx, row in validation_set.iterrows()
    }

    ga_config = {
        "generations": 10,
        "metric": "accuracy",
        "population_config": {
            "population_size": 70,
            "mutation_rate": 0.3,
            "mutate_individually": True,
            "k_elitism": 15,
            "tournament_size": 20,
        },
    }

    cbr = CBR(
        case_base=case_base,
        validation_case_base=validation_case_base,
        k_cases=5,
        config=config,
        use_ga_optimizer=use_ga_optimizer,
        ga_config=ga_config,
        pooling_weights=pooling_weights,
        run_dict=runs_json[run_name],
    )

    runs_json[run_name]["description"] = description
    runs_json[run_name]["stratified"] = stratified
    runs_json[run_name]["use_ga_optimizer"] = use_ga_optimizer

    if use_ga_optimizer:
        runs_json[run_name]["ga_config"] = ga_config

    evaluate_performance(cbr, test_set, runs_json, run_name)

    print("\n")
    print(cbr.pooling_weights)


if __name__ == "__main__":
    main()
