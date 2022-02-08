import sys
import pandas as pd
from utils import config_loader


def validate_ds(features):
    if features.shape[1] != 57:
        exit(1)


if __name__ == "__main__":
    config = config_loader.get_config()
    path = config["common"]["paths"]["features"]
    ds = pd.read_csv(f"{path}")
    validate_ds(ds)
