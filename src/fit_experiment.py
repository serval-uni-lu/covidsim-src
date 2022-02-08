import sys, getopt
import pandas as pd
import numpy as np

from helpers.seir_fit import fit_model
from data_importers import jhu_import as ji

sys.path.append("../")

import logging

from helpers.bayesian_fit import fit_model as fit_bayesian
from utils import config_loader
from pandas.core.common import SettingWithCopyWarning

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
from tqdm import tqdm

from joblib import Parallel, delayed

LOGGER = logging.getLogger()


def main(argv):

    all_countries = pd.DataFrame(
        {
            "ConfirmedCases": [],
            "Fatalities": [],
            "R": [],
            "HospitalizedCases": [],
            "CriticalCases": [],
            "Date": [],
            "CountryName": [],
        }
    )

    # Load config
    config = config_loader.get_config()
    dataset_path = config["common"]["paths"]["features"]
    output_seirhcd_path = config["common"]["paths"]["seirhcd"]

    # Load data
    dataset = ji.prepare_cases_deaths()

    LOGGER.info(dataset.shape)

    # Drop the first col
    # dataset = dataset.drop(["Unnamed: 0"], axis=1)

    # d = (
    #     dataset[["CountryName", "population"]]
    #     .groupby("CountryName")
    #     .min()["population"]
    # )
    # d.is_copy = False
    # populations = dict(zip(list(d.index), d.values))
    d = dataset
    countries = dataset["CountryName"].value_counts().index
    if config["common"]["countries"]["allow"] != "*":
        countries = countries[countries.isin(config["common"]["countries"]["allow"])].to_list()
    else:
        feature_countries = pd.read_csv(dataset_path)["CountryName"].value_counts().index
        countries = countries[countries.isin(feature_countries.to_list())].to_list()

    countries = [c for c in countries if c not in config["common"]["countries"]["deny"]]

    if "ConfirmedCases" not in dataset.columns:
        dataset["ConfirmedCases"] = dataset["ConfirmedCases_y"]

    print(len(countries))

    def single_country(c):
        logging.basicConfig(level=logging.INFO)
        warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
        warnings.simplefilter(action="ignore", category=FutureWarning)
        logging.getLogger().info(f"{c}: Starting.")
        d = dataset[dataset["CountryName"] == c]
        start_date = pd.Timestamp("2020-01-01")
        d = d[d["Date"] > start_date]
        # out = fit_model(c, d, populations.get(c), make_plot=False)
        out = fit_bayesian(c, d)
        # if out2 is not None:
        #     # logging.getLogger().info(f"{c}: Not None.")
        #     R_ = out.loc[split:, "R"] + out2["R"]
        #     out.loc[split:, "R"] = R_.pow(0.5)
        #     out["R"] = out["R"].rolling(7).mean()
        # out["R"] = out["R"].rolling(7, min_periods=1, center=True).mean()
        # if out["R"].isna().sum() > 0:
        #     print("Error")
        logging.getLogger().info(f"{c}: Done.")
        # Temporary test pass
        fake = [
            "InfectiousCases",
            "ExposedCases",
            "RecoveredCases",
            "CriticalCases",
            "HospitalizedCases",
            "Fatalities",
            "c",
            "f",
            "m",
            "t_crit",
            "t_hosp",
        ]
        for i in fake:
            out[i] = np.zeros(out.shape[0])

        out = out[
            [
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
            ]
        ]
        return out

    # countries.remove('Brazil')
    # all_countries_list = Parallel(n_jobs=-1)(
    #     delayed(single_country)(c) for c in tqdm(countries)
    # )
    all_countries_list = [single_country(c) for c in tqdm(countries)]
    logging.getLogger().info(f"All countries: Done.")

    for out in all_countries_list:
        all_countries = all_countries.append(out)

    all_countries.to_csv(output_seirhcd_path)


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    main(sys.argv[1:])
