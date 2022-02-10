import yaml
import sys

DEFAULT_CONFIG = "./config/config.yaml"
DEV_CONFIG = "./config/config-dev.yaml"
ENV = "prod"


def get_config(config_path=None):
    config_path = DEV_CONFIG
    if config_path is None and ENV is not "dev":
        config_path = DEFAULT_CONFIG

    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    return config
