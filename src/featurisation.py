import logging
from utils.config_loader import get_config
import pandas as pd

LOGGER = logging.getLogger()
CONFIG = get_config()


def run():
    logging.basicConfig()
    LOGGER.setLevel(logging.INFO)
    LOGGER.info("Start...")
    LOGGER.info("Loading data...")
    oxford = pd.read_csv(
        CONFIG["common"]["datasets"]["oxford"]["processed"], low_memory=False
    )

    gmobility = pd.read_csv(
        CONFIG["common"]["datasets"]["gmobility"]["processed"], low_memory=False
    )

    rt_estimation = pd.read_csv(
        CONFIG["common"]["datasets"]["rt_estimation"]["processed"], low_memory=False
    )

    demographic = pd.read_csv(
        CONFIG["common"]["datasets"]["demographic"]["processed"], low_memory=False
    )

    country_metrics = pd.read_csv(
        CONFIG["common"]["datasets"]["country_metrics"]["processed"], low_memory=False
    )

    one_hot_feature = ["region"]
    LOGGER.info(f"One hot encoding of {one_hot_feature}")
    country_metrics = pd.get_dummies(country_metrics, columns=one_hot_feature)

    LOGGER.info("Merging oxford and google mobility...")
    out = pd.merge(oxford, gmobility, how="inner", on=["CountryName", "Date"])

    LOGGER.info("Merging Rt estimations...")
    out = pd.merge(out, rt_estimation, how="inner", on=["CountryName", "Date"])

    LOGGER.info("Merging demographic...")
    out = pd.merge(out, demographic, how="inner", on=["CountryName"])

    LOGGER.info("Merging country metrics...")
    out = pd.merge(out, country_metrics, how="inner", on=["CountryName"])

    # CountryName and Date are not removed only for special split.
    to_drop = ["R_max", "R_min"]
    LOGGER.info(f"Dropping columns {to_drop}...")
    out = out.drop(columns=to_drop)

    LOGGER.info(f"Dropping nan... THIS MUST BE LAST STEP.")
    out = out.dropna()

    LOGGER.info("Saving...")
    out.to_csv(CONFIG["common"]["datasets"]["features"]["processed"], index=False)


if __name__ == "__main__":
    run()
