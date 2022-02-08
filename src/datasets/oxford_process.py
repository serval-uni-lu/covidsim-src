import logging
from covidsim_model.src.utils.config_loader import get_config
import pandas as pd
from covidsim_model.src.utils.features import add_days_granularity

LOGGER = logging.getLogger()


def __load_xls(path, selected_measures) -> dict:
    """
    Loads the corresponding excel oxford_file and returns dict of {measure_name : corresponding_df}
    :return:
    """
    xls_oxford = pd.ExcelFile(path)
    res = {}
    for (df_name, s_name) in selected_measures:
        df = pd.read_excel(xls_oxford, f"{s_name}")
        df.index = df.CountryName
        df = df.drop(["CountryCode", "CountryName"], axis=1)
        df = df.transpose()
        res[f"{df_name}"] = df

    return res


"""
Method to join all measures value per country per dates returns dataframe (dates x (countrynames,measures))
:return:
"""


def __normalize(unormalized_measures, selected_measures):
    normalized_measures = unormalized_measures.copy()
    for (measure_name, _) in selected_measures:
        max_value = normalized_measures[measure_name].max()
        factor = -int(100 / max_value)
        normalized_measures[measure_name] *= factor

    return normalized_measures


def run(config_path):
    CONFIG = get_config(config_path)
    logging.basicConfig()
    LOGGER.setLevel(logging.INFO)
    LOGGER.info("Start...")
    selected_measures = [
        ("school", "c1_schoolclosing"),
        ("public_transport", "c5_closepublictransport"),
        ("international_transport", "c8_internationaltravel"),
    ]

    measures_dict = __load_xls(
        CONFIG["common"]["datasets"]["oxford"]["raw"], selected_measures
    )
    measure_name, _ = selected_measures[0]
    measure = measures_dict.get(measure_name)
    dates = measure.index.values
    dates = [pd.to_datetime(d, format="%d%b%Y") for d in dates]
    countries = list(measure.columns)
    d = {"Date": [], "CountryName": []}

    for (n, _) in selected_measures:
        d[f"{n}"]: []

    oxford_measures = pd.DataFrame(d)

    for c in countries:
        country = [c] * len(measure)
        measures_df = []
        columns = [
            "Date",
            "CountryName",
        ]
        for key, val in measures_dict.items():
            measures_df.append(val[c].values)
            columns.append(f"{key}")

        df = list(zip(dates, country, *measures_df))
        df = pd.DataFrame(
            df,
            columns=columns,
        )
        oxford_measures = pd.concat([oxford_measures,df],ignore_index=True)
#        oxford_measures = oxford_measures.append(df, ignore_index=True)

    oxford_measures = __normalize(oxford_measures, selected_measures)
    oxford_measures = add_days_granularity(oxford_measures, ["school", "public_transport", "international_transport"])


    LOGGER.info("Saving...")
    oxford_measures.to_csv(CONFIG["common"]["datasets"]["oxford"]["processed"], index=False)
    LOGGER.info("Done.")


if __name__ == "__main__":
    run()
