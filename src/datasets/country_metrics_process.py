import logging
from src.utils.config_loader import get_config
import pandas as pd
import numpy as np
from functools import reduce

LOGGER = logging.getLogger()


def run(config_path=None):
    CONFIG = get_config(config_path)
    logging.basicConfig()
    LOGGER.setLevel(logging.INFO)
    LOGGER.info("Start...")

    infos = pd.read_csv(CONFIG["common"]["datasets"]["country_metrics"]["raw"])
    infos = infos.replace("Korea, South", "South Korea")
    infos["Country"] = infos["Country"].str.strip()

    regions_indices = infos.Region.unique()
    infos["regions_indices"] = infos["Region"].replace(
        regions_indices, np.arange(len(regions_indices))
    )
    infos["CountryName"] = infos["Country"]
    filter_columns = [
        "CountryName",
        "regions_indices",
        "GDP ($ per capita)",
        "Area (sq. mi.)",
    ]
    infos = infos[filter_columns]
    infos.columns = [
        "CountryName",
        "region",
        "gdp",
        "area",
    ]
    LOGGER.info("Saving...")
    infos.to_csv(
        CONFIG["common"]["datasets"]["country_metrics"]["processed"], index=False
    )
    LOGGER.info("Done.")


if __name__ == "__main__":
    run()
