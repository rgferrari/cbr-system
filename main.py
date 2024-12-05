import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from cbr import CBR
from dataset import heart_disease


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

    print('\nEvaluation:')
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")


def main():
    train_set, test_set, config, pooling_weights = heart_disease()

    case_base = {idx: row.to_dict() for idx, row in train_set.iterrows()}

    ga_config = {
        "generations": 10,
        "metric": "accuracy",
        "population_config": {
            "population_size": 70,
            "mutation_rate": 1,
            "mutate_individually": True,
            "k_elitism": 10,
            "tournament_size": 15,
        },
    }

    cbr = CBR(case_base, 
              config, 
              use_ga_optimizer=False, 
              ga_config=ga_config,
              pooling_weights=pooling_weights)

    evaluate_performance(cbr, test_set)

if __name__ == "__main__":
    main()
