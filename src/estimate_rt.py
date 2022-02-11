import logging

from joblib import Parallel, delayed

from src.helpers.bayesian_fit import fit_model as fit_bayesian

from pandas.core.common import SettingWithCopyWarning
from tqdm import tqdm

from src.utils.config_loader import get_config
import pandas as pd

LOGGER = logging.getLogger()
CONFIG = get_config()
import warnings


def run():
    logging.basicConfig()
    LOGGER.setLevel(logging.INFO)
    LOGGER.info("Start...")

    LOGGER.info("Loading data...")
    cases = pd.read_csv(
        CONFIG["common"]["datasets"]["cases_death"]["processed"],
        parse_dates=["Date"],
        low_memory=False,
    )

    countries = cases["CountryName"].value_counts().index
    # Filter allowed country
    if CONFIG["estimate_rt"]["countries"]["allow"] != "*":
        countries = countries[
            countries.isin(CONFIG["estimate_rt"]["countries"]["allow"])
        ].to_list()

    # Filter denied country
    countries = [c for c in countries if c not in CONFIG["estimate_rt"]["countries"]["deny"]]

    LOGGER.info(f"Estimate Rt for {len(countries)} countries.")

    def single_country(c):
        logging.basicConfig(level=logging.INFO)
        warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
        warnings.simplefilter(action="ignore", category=FutureWarning)
        logging.getLogger().info(f"{c}: Starting...")
        d = cases[cases["CountryName"] == c]
        out = fit_bayesian(c, d)

        logging.getLogger().info(f"{c}: Done.")
        return out

    # Multithread
    # countries.remove('Brazil')
    # all_countries_list = Parallel(n_jobs=-1)(
    #     delayed(single_country)(c) for c in tqdm(countries)
    # )
    # Single thread
    all_countries_list = [single_country(c) for c in tqdm(countries)]
    logging.getLogger().info(f"All countries: Done.")

    all_countries = pd.concat(all_countries_list).reindex()
    filter_columns = ["CountryName", "Date", "R", "R_min", "R_max"]
    all_countries = all_countries[filter_columns]

    LOGGER.info("Saving...")
    all_countries.to_csv(
        CONFIG["common"]["datasets"]["rt_estimation"]["processed"], index=False
    )
    LOGGER.info("Done.")


if __name__ == "__main__":
    run()
