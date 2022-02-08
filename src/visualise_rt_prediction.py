import logging
from pathlib import Path

import numpy as np
from helpers.bayesian_fit import fit_model
from utils.config_loader import get_config
import pandas as pd
import matplotlib.pyplot as plt
from data_importers.who_import import prepare_cases_deaths
import datatest as dt

LOGGER = logging.getLogger()
CONFIG = get_config()

from utils import filter_allow_deny


def generate(rt_predict, ground_truth):
    fig, ax = plt.subplots()
    ax.plot(rt_predict["Date"], rt_predict["R"], color="red", label="Prediction")
    ax.fill_between(
        x=rt_predict["Date"],
        y1=rt_predict["R_min"],
        y2=rt_predict["R_max"],
        alpha=0.2,
        facecolor="red",
        label="50% confidence interval P",
    )

    ax.plot(
        ground_truth["date"],
        ground_truth["R_t-estimate"],
        color="blue",
        label="Ground truth",
    )
    ax.fill_between(
        x=ground_truth["date"],
        y1=ground_truth["Low_50"],
        y2=ground_truth["High_50"],
        alpha=0.2,
        facecolor="blue",
        label="50% confidence interval GT",
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Rt")
    ax.legend()


def run():
    logging.basicConfig()
    LOGGER.setLevel(logging.INFO)
    LOGGER.info("Starting...")

    Path(CONFIG["visualise_rt_prediction"]["visualisation_dir"]).mkdir(
        parents=True, exist_ok=True
    )
    LOGGER.info("Loading data...")
    rt_predict = pd.read_csv(
        CONFIG["common"]["datasets"]["rt_estimation"]["processed"],
        parse_dates=["Date"],
        low_memory=False,
    )

    for country in CONFIG["visualise_rt_prediction"]["countries"]:
        LOGGER.info(f"{country}...")
        rt_predict_c = rt_predict[rt_predict["CountryName"] == country]
        ground_truth_path = f"{CONFIG['visualise_rt_prediction']['ground_truth_dir']}/{country.lower()}_rt.csv"
        ground_truth = pd.read_csv(
            ground_truth_path, parse_dates=["date"], low_memory=False
        )
        generate(rt_predict_c, ground_truth)
        figure_path = f"{CONFIG['visualise_rt_prediction']['visualisation_dir']}/{country.lower()}.pdf"

        plt.savefig(figure_path)

    LOGGER.info("Done.")


if __name__ == "__main__":
    run()
