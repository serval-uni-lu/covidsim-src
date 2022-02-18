import pandas as pd
import os
import logging
from src.utils.config_loader import get_config

LOGGER = logging.getLogger()
CONFIG = get_config()
import warnings

def _get_csv_as_dataframe(name):
    return pd.read_csv(f"{CONFIG['common']['datasets'][name]['processed']}",low_memory=False)


def run():
    oxford = _get_csv_as_dataframe( 'oxford')

    gmobility = _get_csv_as_dataframe( 'gmobility')

    demographic = _get_csv_as_dataframe( 'demographic')
    rt_estimation = _get_csv_as_dataframe( 'rt_estimation')
    country_metrics = _get_csv_as_dataframe( 'country_metrics')
    country_metrics = pd.get_dummies(country_metrics,columns=['region'])

    out = pd.merge(oxford, gmobility, how="inner", on=["CountryName", "Date"])
    out = pd.merge(out, rt_estimation, how="inner", on=["CountryName", "Date"])
    out = pd.merge(out, demographic, how="inner", on=["CountryName"])
    out = pd.merge(out, country_metrics, how="inner", on=["CountryName"])
    out = out.drop(columns=["R_max","R_min"])
    out = out.dropna()
    merged_data_path = CONFIG['common']['datasets']['ml']['dataset']

    out.to_csv(merged_data_path, index=False)
    
if __name__ == "__main__":
    LOGGER.setLevel(logging.INFO)
    LOGGER.info("Start...")
    run()
    LOGGER.info("Merge done")
