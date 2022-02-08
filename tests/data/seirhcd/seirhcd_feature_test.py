from src.utils.config_loader import get_config
import pytest
import pandas as pd
import datatest as dt
import logging

pytestmark = pytest.mark.filterwarnings("ignore:subset and superset warning")
LOGGER = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def df() -> pd.DataFrame:
    return pd.read_csv(
        get_config("./config/config.yaml")["common"]["paths"]["seirhcd"],
        low_memory=False,
    )  # Returns DataFrame.


def test_column_names(df):
    required_names = {
        "Unnamed: 0",
        "ConfirmedCases",
        "Fatalities",
        "R",
        "HospitalizedCases",
        "CriticalCases",
        "Date",
        "CountryName",
        "ExposedCases",
        "RecoveredCases",
        "InfectiousCases",
        "t_hosp",
        "t_crit",
        "m",
        "c",
        "f",
        "R_min",
        "R_max"
    }

    dt.validate(df.columns, required_names)
    LOGGER.info(df.shape)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
