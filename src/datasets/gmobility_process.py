import logging
from covidsim_model.src.utils.config_loader import get_config
import pandas as pd
from covidsim_model.src.utils.features import add_days_granularity

LOGGER = logging.getLogger()


def run(config_path):
    CONFIG = get_config(config_path)
    logging.basicConfig()
    LOGGER.setLevel(logging.INFO)
    LOGGER.info("Start...")

    mobility_csv = pd.read_csv(
        CONFIG["common"]["datasets"]["gmobility"]["raw"], dtype={4: str}
    )

    # Remove Data from sub-region
    mobility_csv = mobility_csv[mobility_csv["sub_region_1"].isna()]

    # Retrieve only subset of data
    mobility = mobility_csv[
        [
            "country_region",
            "date",
            "retail_and_recreation_percent_change_from_baseline",
            "grocery_and_pharmacy_percent_change_from_baseline",
            "parks_percent_change_from_baseline",
            "transit_stations_percent_change_from_baseline",
            "workplaces_percent_change_from_baseline",
            "residential_percent_change_from_baseline",
        ]
    ]

    # Update columns name to fit future data
    mobility.columns = [
        "CountryName",
        "Date",
        "retail/recreation",
        "grocery/pharmacy",
        "parks",
        "transit_stations",
        "workplace",
        "residential",
    ]
    features = [
        "retail/recreation",
        "grocery/pharmacy",
        "parks",
        "transit_stations",
        "workplace",
        "residential",
    ]

    mobility[features] = mobility[features].clip(-100, 100)
    add_days_granularity(mobility, features)

    LOGGER.info("Saving...")
    mobility.to_csv(CONFIG["common"]["datasets"]["gmobility"]["processed"], index=False)
    LOGGER.info("Done.")


if __name__ == "__main__":
    run()
