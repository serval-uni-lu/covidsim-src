from src.utils.config_loader import get_config
import pytest
import pandas as pd
import datatest as dt
import logging

LOGGER = logging.getLogger(__name__)
pytestmark = pytest.mark.filterwarnings("ignore:subset and superset warning")


@pytest.fixture(scope="session")
def df() -> pd.DataFrame:
    return pd.read_csv(
        get_config("./config/config.yaml")["common"]["datasets"]["features"][
            "processed"
        ],
        low_memory=False,
    )  # Returns DataFrame.


def test_column_names(df):
    required_names = {
        "Date",
        "CountryName",
        "parks",
        "residential",
        "retail/recreation",
        "transit_stations",
        "workplace",
        "parks_5days",
        "residential_5days",
        "retail/recreation_5days",
        "transit_stations_5days",
        "workplace_5days",
        "parks_10days",
        "residential_10days",
        "retail/recreation_10days",
        "transit_stations_10days",
        "workplace_10days",
        "parks_15days",
        "residential_15days",
        "retail/recreation_15days",
        "transit_stations_15days",
        "workplace_15days",
        "parks_30days",
        "residential_30days",
        "retail/recreation_30days",
        "transit_stations_30days",
        "workplace_30days",
        "grocery/pharmacy",
        "grocery/pharmacy_10days",
        "grocery/pharmacy_15days",
        "grocery/pharmacy_30days",
        "grocery/pharmacy_5days",
        "international_transport",
        "international_transport_10days",
        "international_transport_15days",
        "international_transport_30days",
        "international_transport_5days",
        "public_transport",
        "public_transport_10days",
        "public_transport_15days",
        "public_transport_30days",
        "public_transport_5days",
        "school",
        "school_10days",
        "school_15days",
        "school_30days",
        "school_5days",
        "region_0",
        "region_1",
        "region_2",
        "region_3",
        "region_4",
        "region_5",
        "region_6",
        "region_7",
        "region_8",
        "region_10",
        "region_9",
        "density",
        "demographic",
        "population_p14",
        "population_p65",
        "gdp",
        "area",
        "R",
    }
    dt.validate(df.columns, required_names)


def test_no_nan(df: pd.DataFrame):
    assert df.shape == df.dropna().shape


def test_not_empty(df: pd.DataFrame):
    assert df.shape[0] > 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
