import pandas as pd
from sklearn.datasets import load_iris

from cbr import CBR


def main():
    iris = load_iris()
    iris_data = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_data['target'] = iris.target

    case_base = {
        idx: row.to_dict() for idx, row in iris_data.iterrows()
    }

    config = {
        "sepal length (cm)": "numeric",
        "sepal width (cm)": "numeric",
        "petal length (cm)": "numeric",
        "petal width (cm)": "numeric",
    }

    cbr = CBR(case_base, config)

    query = {
        "sepal length (cm)": 5.1,
        "sepal width (cm)": 3.5,
        "petal length (cm)": 1.4,
        "petal width (cm)": 0.2,
    }

    result = cbr.retrieve(query)

    print("Retrieved case:", result)


if __name__ == "__main__":
    main()

# feature_max = iris_data.max()

# numeric_similarity = {
#     "sepal length (cm)": sim.numbers.linear(max=feature_max["sepal length (cm)"]),
#     "sepal width (cm)": sim.numbers.linear(max=feature_max["sepal width (cm)"]),
#     "petal length (cm)": sim.numbers.linear(max=feature_max["petal length (cm)"]),
#     "petal width (cm)": sim.numbers.linear(max=feature_max["petal width (cm)"]),
# }

# weights = {
#     "sepal length (cm)": 0.25,
#     "sepal width (cm)": 0.25,
#     "petal length (cm)": 0.25,
#     "petal width (cm)": 0.25,
# }

# global_similarity = sim.attribute_value(
#     attributes={
#         feature: numeric_similarity[feature] for feature in numeric_similarity
#     }
# )

# retriever = retrieval.build(global_similarity, limit=1)

# case_base = {
#     idx: row.to_dict() for idx, row in iris_data.iterrows()
# }

# query = {
#     "sepal length (cm)": 5.1,
#     "sepal width (cm)": 3.5,
#     "petal length (cm)": 1.4,
#     "petal width (cm)": 0.2,
# }

# results = retrieval.apply(case_base, query, retriever)

# print("Retrieved cases:", results)