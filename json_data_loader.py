import json
import numpy as np

json_train = "C:\json\data_train.json"
json_test = "C:\json\data_test.json"
json_valid = "C:\json\data_valid.json"


def load_json_data():
    with open(json_train, "r") as fp:
        data_train = json.load(fp)
    with open(json_valid, "r") as fp:
        data_valid = json.load(fp)
    with open(json_test, "r") as fp:
        data_test = json.load(fp)
    X_train = np.array(data_train["X_training"])
    y_train = np.array(data_train["y_training"])
    X_validation = np.array(data_valid["X_validation"])
    y_validation = np.array(data_valid["y_validation"])
    X_test = np.array(data_test["X_testing"])
    y_test = np.array(data_test["y_testing"])
    return (X_train, X_validation, X_test, y_train, y_validation, y_test)


def load_test_data():
    with open(json_test, "r") as fp:
        data_test = json.load(fp)
    X_test = np.array(data_test["X_testing"])
    y_test = np.array(data_test["y_testing"])
    return X_test, y_test