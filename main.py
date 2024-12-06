import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from cbr import CBR
from dataset import heart_disease
import json
import numpy as np
from datetime import datetime


def evaluate_performance(
    cbr,
    test_set,
    runs_json=None,
    run_name=None,
    retrieved_cases=5,
    runs_filename="runs.json",
):
    y_true = []
    y_pred = []

    for idx, row in test_set.iterrows():
        query = row.to_dict()
        true_label = query.pop("target")
        result = cbr.retrieve(query)
        similar_cases = result.ranking[:retrieved_cases]
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

        with open(runs_filename, "w") as f:
            json.dump(runs_json, f)


def main():
    scheduler = json.load(open("scheduler.json", "r"))

    timestamp = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

    runs_json = {}
    runs_filename = f"runs_{timestamp}.json"

    for run in scheduler["runs"]:
        run_name = run["name"]
        description = run["description"]
        stratified = run["stratified"]
        use_ga_optimizer = run["use_ga_optimizer"]
        retrieved_cases = run["retrieved_cases"]
        ga_config = run.get("ga_config", None)
        initial_weights = run.get("initial_weights", None)

        runs_json[run_name] = {}

        train_set, validation_set, test_set, config, pooling_weights = heart_disease(
            validation=True, stratify=stratified
        )

        case_base = {idx: row.to_dict() for idx, row in train_set.iterrows()}

        validation_case_base = {
            idx: row.to_dict() for idx, row in validation_set.iterrows()
        }

        if initial_weights is not None:
            if isinstance(initial_weights, float) or isinstance(initial_weights, int):
                pooling_weights = {
                    feature: initial_weights for feature in pooling_weights
                }
        else:
            pooling_weights = None

        cbr = CBR(
            case_base=case_base,
            validation_case_base=validation_case_base,
            k_cases=retrieved_cases,
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

        evaluate_performance(
            cbr,
            test_set,
            runs_json,
            run_name,
            retrieved_cases=retrieved_cases,
            runs_filename=runs_filename,
        )

        print("\n")
        print(cbr.pooling_weights)


if __name__ == "__main__":
    main()
