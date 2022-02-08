import logging
from src.utils.config_loader import get_config
from data_importers.who_import import prepare_cases_deaths
import matplotlib.pyplot as plt
import pandas as pd

LOGGER = logging.getLogger()
CONFIG = get_config()


def clean_cumulative(df):
    df["New_cases"][df["New_cases"] < 0] = 0
    df["ConfirmedCases"] = df["New_cases"].cumsum()
    return df


def run():
    data = prepare_cases_deaths()
    data = data[data["CountryName"] == "Luxembourg"]
    print(data.shape)
    WHO_PATH = "./data/raw/who/WHO-COVID-19-global-data.csv"
    cases_deaths = pd.read_csv(WHO_PATH, parse_dates=["Date_reported"])

    # df = cases_deaths.copy()
    # cases_deaths = cases_deaths[cases_deaths["Country"] == "Luxembourg"]
    # cases_deaths["ConfirmedCases"] = cases_deaths["New_cases"].cumsum()
    # cases_deaths.plot(x="Date_reported", y="ConfirmedCases")
    # cases_deaths["New_cases"][cases_deaths["New_cases"] < 0] = 0
    # cases_deaths["ConfirmedCases"] = cases_deaths["New_cases"].cumsum()
    # cases_deaths.plot(x="Date_reported", y="ConfirmedCases")
    data.plot(x="Date", y="ConfirmedCases")

    # print((cases_deaths["New_cases"] >= 0).sum())
    plt.show()


if __name__ == "__main__":
    run()
