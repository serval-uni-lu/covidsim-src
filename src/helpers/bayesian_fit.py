# -*- coding: utf-8 -*-
"""
# Estimating COVID-19's $R_t$
*Based on the work of Kevin Systrom - https://github.com/k-sys/covid-19/blob/master/Realtime%20R0.ipynb*
"""

import pandas as pd
import numpy as np

from scipy import stats as sps
from matplotlib.dates import date2num, num2date
import datetime as DT
# We create an array for every possible value of Rt
R_T_MAX = 10
r_t_range = np.linspace(0, R_T_MAX, R_T_MAX * 100 + 1)
# Gamma is 1/serial interval
# https://wwwnc.cdc.gov/eid/article/26/7/20-0282_article
# https://www.nejm.org/doi/full/10.1056/NEJMoa2001316
GAMMA = 1 / 3
rng = np.random.RandomState(2021)


def highest_density_interval(pmf, p=.9, debug=False):
    # If we pass a DataFrame, just call this recursively on the columns
    if (isinstance(pmf, pd.DataFrame)):
        return pd.DataFrame([highest_density_interval(pmf[col], p=p) for col in pmf],
                            index=pmf.columns)

    cumsum = np.cumsum(pmf.values)

    # N x N matrix of total probability mass for each low, high
    total_p = cumsum - cumsum[:, None]

    # Return all indices with total_p > p
    lows, highs = (total_p > p).nonzero()

    # Check to return Nan if the lows and highs can not be calculated because maximum likelihood Rt is < p
    if len(lows) is 0 :
        return pd.Series([np.NaN, np.NaN],
                     index=[f'Low_{p * 100:.0f}',
                            f'High_{p * 100:.0f}'])

    # Find the smallest range (highest density)
    best = (highs - lows).argmin()

    low = pmf.index[lows[best]]
    high = pmf.index[highs[best]]

    # print(f"{low}::{high}")
    return pd.Series([low, high],
                     index=[f'Low_{p * 100:.0f}',
                            f'High_{p * 100:.0f}'])


def prepare_cases(cases, cutoff=25):
    new_cases = cases.diff()

    smoothed = new_cases.rolling(7,  # 7 days moving window for smoothing
                                 # win_type='gaussian',   #or comment whole line to have uniform
                                 min_periods=1,
                                 center=True).mean(std=2).round()
    idx_start = np.argwhere(smoothed.to_numpy() == 0)

    if idx_start.shape[0] == 0:
        idx_start = 1
    else:
        idx_start = idx_start.max() + 1

    smoothed = smoothed.iloc[idx_start:]
    original = new_cases.loc[smoothed.index]

    return original, smoothed


def get_posteriors(sr, date, sigma=0.15):
    # (1) Calculate Lambda

    gamma = 1 / rng.normal(4, 0.2, len(r_t_range))
    lam = sr[:-1] * np.exp(gamma[:, None] * (r_t_range[:, None] - 1))
    # lam = sr[:-1] * np.exp(GAMMA * (r_t_range[:, None] - 1))

    # (2) Calculate each day's likelihood
    likelihoods = pd.DataFrame(
        data=sps.poisson.pmf(sr[1:], lam),
        index=r_t_range,
        columns=date[1:])

    # (3) Create the Gaussian Matrix
    process_matrix = sps.norm(loc=r_t_range,
                              scale=sigma
                              ).pdf(r_t_range[:, None])

    # (3a) Normalize all rows to sum to 1
    process_matrix /= process_matrix.sum(axis=0)

    # (4) Calculate the initial prior
    # prior0 = sps.gamma(a=4).pdf(r_t_range)
    prior0 = np.ones_like(r_t_range) / len(r_t_range)
    prior0 /= prior0.sum()

    # Create a DataFrame that will hold our posteriors for each day
    # Insert our prior as the first posterior.
    posteriors = pd.DataFrame(
        index=r_t_range,
        columns=date,
        data={date[0]: prior0}
    )

    # We said we'd keep track of the sum of the log of the probability
    # of the data for maximum likelihood calculation.
    log_likelihood = 0.0

    # (5) Iteratively apply Bayes' rule
    for previous_day, current_day in zip(date[:-1], date[1:]):
        # (5a) Calculate the new prior
        current_prior = process_matrix @ posteriors[previous_day]

        # (5b) Calculate the numerator of Bayes' Rule: P(k|R_t)P(R_t)
        numerator = likelihoods[current_day] * current_prior

        # (5c) Calcluate the denominator of Bayes' Rule P(k)
        denominator = np.sum(numerator)

        if denominator == 0:
            print(0)

        # Execute full Bayes' Rule
        posteriors[current_day] = numerator / denominator

        # Add to the running sum of log likelihoods
        log_likelihood += np.log(denominator)

    return posteriors, log_likelihood


def fit_model(area_name, df, R_min=0.2):
    countries_output = {}
    cases_columns = "ConfirmedCases"

    train_data = df[(df["CountryName"] == area_name) & (df[cases_columns] > 0)]

    cases = train_data[cases_columns]

    original, smoothed = prepare_cases(cases)
    original_array = original.values
    smoothed_array = smoothed.values

    if smoothed_array.shape[0] == 0:
        return None

    dates = smoothed.index
    # dates_detection = date2num(smoothed.index.tolist())
    # dates_infection = smoothed.index - DT.timedelta(days=9)
    # dates_infection = date2num(dates_infection.tolist())

    posteriors, log_likelihood = get_posteriors(smoothed_array, dates, sigma=.15)
    hdis = highest_density_interval(posteriors, p=.5)
    most_likely = posteriors.idxmax().rename('R_t-estimate')
    result = pd.concat([most_likely, hdis], axis=1)

    train_data["R"] = result["R_t-estimate"]
    train_data["R"] = train_data["R"].rolling(3, min_periods=1).mean()
    train_data["R_min"] = result["Low_50"].rolling(3, min_periods=1).mean()
    train_data["R_max"] = result["High_50"].rolling(3, min_periods=1).mean()
    return train_data


if __name__ == "__main__":
    dataset_path = "./dataset/features.csv"
    dataset = pd.read_csv(dataset_path, parse_dates=["Date"])
    dataset = dataset.drop(["Unnamed: 0"], axis=1)

    if "ConfirmedCases" not in dataset.columns:
        dataset["ConfirmedCases"] = dataset["ConfirmedCases_y"]

    lux = fit_model("Luxembourg", dataset)
