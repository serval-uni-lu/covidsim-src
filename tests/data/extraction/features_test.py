from src.utils.config_loader import get_config
import pytest
import pandas as pd
import datatest as dt
from datetime import date, timedelta
import logging

pytestmark = pytest.mark.filterwarnings("ignore:subset and superset warning")

LOGGER = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def df() -> pd.DataFrame:
    return pd.read_csv(
        get_config("./config/config.yaml")["common"]["paths"]["features"],
        parse_dates=["Date"],
        low_memory=False,
    )  # Returns DataFrame.


def test_column_names(df):
    required_names = {
        "Unnamed: 0",
        "CountryName",
        "Date",
        "retail/recreation",
        "grocery/pharmacy",
        "parks",
        "transit_stations",
        "workplace",
        "residential",
        "school",
        "public_transport",
        "international_transport",
        "ConfirmedCases",
        "ConfirmedDeaths",
        "density",
        "population",
        "population_p65",
        "population_p14",
        "gdp",
        "area",
        "region",
        "retail/recreation_15days",
        "retail/recreation_10days",
        "retail/recreation_5days",
        "retail/recreation_30days",
        "grocery/pharmacy_15days",
        "grocery/pharmacy_10days",
        "grocery/pharmacy_5days",
        "grocery/pharmacy_30days",
        "parks_15days",
        "parks_10days",
        "parks_5days",
        "parks_30days",
        "transit_stations_15days",
        "transit_stations_10days",
        "transit_stations_5days",
        "transit_stations_30days",
        "workplace_15days",
        "workplace_10days",
        "workplace_5days",
        "workplace_30days",
        "residential_15days",
        "residential_10days",
        "residential_5days",
        "residential_30days",
        "school_15days",
        "school_10days",
        "school_5days",
        "school_30days",
        "public_transport_15days",
        "public_transport_10days",
        "public_transport_5days",
        "public_transport_30days",
        "international_transport_15days",
        "international_transport_10days",
        "international_transport_5days",
        "international_transport_30days",
    }

    dt.validate(df.columns, required_names)


def test_not_outdated(df):
    one_month_ago = pd.to_datetime("now") - pd.Timedelta(31, "days")
    dt.validate.interval(
        df["Date"].max(), min=one_month_ago, msg="one month old or newer"
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
