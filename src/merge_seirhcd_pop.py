import logging
from utils.config_loader import get_config
from utils import load_dataset
LOGGER = logging.getLogger()
CONFIG = get_config()


def run():
    df = load_dataset()
    df = df.dropna()
    df.to_csv(CONFIG["common"]["paths"]["ml_features"])


if __name__ == "__main__":
    run()
