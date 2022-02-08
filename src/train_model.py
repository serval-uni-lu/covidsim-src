import json
import logging
from copy import deepcopy

import joblib
import yaml
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import utils
from utils.config_loader import get_config
import numpy as np

LOGGER = logging.getLogger()
CONFIG = get_config()


def scale_data(X_train, X_test):
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return scaler, X_train_scaled, X_test_scaled


def find_best_bayesian_ridge(X_train, y_train):
    reg = BayesianRidge(compute_score=True, tol=1e-5)
    parameters = {
        "alpha_init": (0.2, 0.5, 1, 1.5),
        "lambda_init": [1e-3, 1e-4, 1e-5, 1e-6],
    }
    srch = GridSearchCV(reg, parameters)
    srch.fit(X_train, y_train)
    params = srch.get_params()

    reg.set_params(
        alpha_init=params["estimator__alpha_init"],
        lambda_init=params["estimator__lambda_init"],
    )
    reg.fit(X_train, y_train)

    return reg, params


def find_best_model(X_train, y_train, X_test, y_test):
    best_perf = -1
    best_model = None
    best_reports = None

    for i in range(CONFIG["train_model"]["iterations"]):
        LOGGER.info("Iter search {}".format(i))
        parameter_space = {
            "hidden_layer_sizes": [
                (1000, 50),
                (50, 100, 50),
                (50, 100, 100),
                (50, 500, 50),
            ],
            "alpha": [0.0001, 0.05],
        }

        mlp = MLPRegressor((1000, 50), max_iter=1500, verbose=False, solver="adam")
        mlp_clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=CONFIG["train_model"]["cross_validation"], verbose=True)
        mlp_clf.fit(X_train, y_train)

        reports, __ = (
            utils.metrics_report(X_test, y_test, mlp_clf),
            utils.metrics_report(X_train, y_train, mlp_clf),
        )

        if reports["r2_score"] > best_perf:
            best_model = deepcopy(mlp_clf.best_estimator_ if mlp != mlp_clf else mlp)
            best_perf = reports["r2_score"]
            best_reports = reports

    return best_model, best_reports


def run():
    logging.basicConfig()
    LOGGER.setLevel(logging.INFO)
    LOGGER.info("Starting...")
    LOGGER.info("Loading data...")
    ml_path = CONFIG["common"]["datasets"]["ml"]["processed"]
    X_train = np.load(f"{ml_path}/X_train.npy")
    X_test = np.load(f"{ml_path}/X_test.npy")
    y_train = np.load(f"{ml_path}/y_train.npy")
    y_test = np.load(f"{ml_path}/y_test.npy")

    LOGGER.info("Scaling data...")
    scaler, X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    LOGGER.info("Searching best model...")
    model, reports = find_best_model(X_train_scaled, y_train, X_test_scaled, y_test)

    LOGGER.info("Searching best bayesian_ridge...")
    reg, __ = find_best_bayesian_ridge(X_train, y_train)
    __, y_std = reg.predict(X_test, return_std=True)

    LOGGER.info("Saving...")
    joblib.dump(model, CONFIG["train_model"]["paths"]["model"])
    joblib.dump(scaler, CONFIG["train_model"]["paths"]["scaler"])
    with open(CONFIG["train_model"]["paths"]["report"], "w") as fp:
        json.dump(
            {
                "perf": reports,
                "std_test": list(y_std),
                "hidden_layer_sizes": model.hidden_layer_sizes,
            },
            fp,
            indent=4
        )
    LOGGER.info("Done...")


if __name__ == "__main__":
    run()
