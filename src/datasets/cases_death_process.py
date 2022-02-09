import logging
from src.utils.config_loader import get_config
import pandas as pd

LOGGER = logging.getLogger()


def __load_jhu(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[~df["Province/State"].notnull()]
    df.index = df["Country/Region"]
    df = df.drop(["Lat", "Long", "Province/State", "Country/Region"], axis=1)
    return df.transpose()


def run(config_path=None):    
    CONFIG = get_config(config_path)
    logging.basicConfig()
    LOGGER.setLevel(logging.INFO)
    LOGGER.info("Start...")

    cases = __load_jhu(CONFIG["common"]["datasets"]["cases_death"]["raw"]["cases"])
    deaths = __load_jhu(CONFIG["common"]["datasets"]["cases_death"]["raw"]["deaths"])
    dates = cases.index.values[2:]
    dates = [pd.to_datetime(d) for d in dates]

    cases_deaths = pd.DataFrame(
        {"Date": [], "CountryName": [], "ConfirmedCases": [], "ConfirmedDeaths": []}
    )
    countries = list(cases.columns)
    for c in countries:
        confirmed_c = cases[c].values
        death_c = deaths[c].values
        country = [c] * len(confirmed_c)
        df = list(zip(dates, country, confirmed_c, death_c))
        df = pd.DataFrame(
            df, columns=["Date", "CountryName", "ConfirmedCases", "ConfirmedDeaths"]
        )
        new_cases = df["ConfirmedCases"].diff()
        new_cases[0] = 0
        new_cases[new_cases < 0] = 0
        df["ConfirmedCases"] = new_cases.cumsum()
        new_death = df["ConfirmedDeaths"].diff()
        new_death[0] = 0
        new_death[new_death < 0] = 0
        df["ConfirmedDeaths"] = new_death.cumsum()
        cases_deaths = pd.concat([cases_deaths,df], ignore_index=True)

    LOGGER.info("Saving...")
    cases_deaths.to_csv(
        CONFIG["common"]["datasets"]["cases_death"]["processed"], index=False
    )
    LOGGER.info("Done.")


if __name__ == "__main__":
    run()
