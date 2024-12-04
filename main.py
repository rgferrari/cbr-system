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

    cbr.print(result)


if __name__ == "__main__":
    main()
