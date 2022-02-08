import logging
from utils.config_loader import get_config
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np

LOGGER = logging.getLogger()
CONFIG = get_config()


def run():
    df = pd.read_csv(
        CONFIG["common"]["datasets"]["features"]["processed"],
        parse_dates=["Date"],
        low_memory=False
    )

    # Retrieve target
    y = df.pop(CONFIG["split_data"]["target"]).to_numpy()

    # Remove CountryName and Date, this feature were kept only for future special splits.
    df = df.drop(columns=["CountryName", "Date"])
    X = df.to_numpy()

    # Random split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=CONFIG["split_data"]["test_size"],
        random_state=CONFIG["split_data"]["seed"],
    )

    # Save
    ml_path = CONFIG["common"]["datasets"]["ml"]["processed"]
    suffix = "npy"
    Path(ml_path).mkdir(parents=True, exist_ok=True)
    np.save(f"{ml_path}/X_train.{suffix}", X_train)
    np.save(f"{ml_path}/X_test.{suffix}", X_test)
    np.save(f"{ml_path}/y_train.{suffix}", y_train)
    np.save(f"{ml_path}/y_test.{suffix}", y_test)


if __name__ == "__main__":
    run()
