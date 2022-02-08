import logging
from covidsim_model.src.utils.config_loader import get_config
import pandas as pd
from functools import reduce

LOGGER = logging.getLogger()


def run(config_path):
    CONFIG = get_config(config_path)
    logging.basicConfig()
    LOGGER.setLevel(logging.INFO)
    LOGGER.info("Start...")

    demographic_info_path = CONFIG["common"]["datasets"]["demographic"]["raw"]

    demographics = []
    demographics_label = [
        "CountryName",
        "density",
        "demographic",
        "population_p65",
        "population_p14",
    ]

    demographics_files = [
        "population_density_long",
        "population_total_long",
        "population_above_age_65_percentage_long",
        "population_below_age_14_percentage_long",
    ]
    for i, e in enumerate(demographics_files):
        f = f"{demographic_info_path}/{e}.csv"
        demo = pd.read_csv(f)
        demo = demo.loc[demo.groupby("Country Name").Year.idxmax()]
        demo = demo[["Country Name", "Count"]]
        demo.columns = ["CountryName", "value"]
        demo = demo.replace("Korea, Rep.", "South Korea")
        demographics.append(demo)

    demographics = reduce(
        lambda left, right: pd.merge(left, right, on=["CountryName"], how="inner"),
        demographics,
    )

    demographics.columns = demographics_label
    LOGGER.info("Saving...")
    demographics.to_csv(
        CONFIG["common"]["datasets"]["demographic"]["processed"], index=False
    )
    LOGGER.info("Done.")


if __name__ == "__main__":
    run()
