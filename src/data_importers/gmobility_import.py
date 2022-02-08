import pandas as pd
import wget

from src.utils import config_loader

config = config_loader.get_config()

mobility_file_path = "./raw_data/Global_Mobility_Report.csv"


def download_mobility_data():
    google_mobility_url = (
        "https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv"
    )
    wget.download(google_mobility_url, mobility_file_path)


def prepare_mobility_features():
    """
    Retrieve google mobility data per countries
    :return: Dataframe containing mobility data per countries
    
    """
    # download_mobility_data()
    mobility_csv = pd.read_csv(
        config["common"]["paths"]["gmobility"], parse_dates=["date"], dtype={4: str}
    )

    # remove sub region data
    mobility_csv = mobility_csv[mobility_csv['sub_region_1'].isna()]

    mobility = mobility_csv[
        [
            "country_region",
            "date",
            "retail_and_recreation_percent_change_from_baseline",
            "grocery_and_pharmacy_percent_change_from_baseline",
            "parks_percent_change_from_baseline",
            "transit_stations_percent_change_from_baseline",
            "workplaces_percent_change_from_baseline",
            "residential_percent_change_from_baseline",
        ]
    ]
    mobility.columns = [
        "CountryName",
        "Date",
        "retail/recreation",
        "grocery/pharmacy",
        "parks",
        "transit_stations",
        "workplace",
        "residential",
    ]

    mobility = mobility.dropna()
    mobility[
        [
            "retail/recreation",
            "grocery/pharmacy",
            "parks",
            "transit_stations",
            "workplace",
            "residential",
        ]
    ] = mobility[
        [
            "retail/recreation",
            "grocery/pharmacy",
            "parks",
            "transit_stations",
            "workplace",
            "residential",
        ]
    ].clip(
        -100, 100
    )

    return mobility

if __name__ == "__main__":
    download_mobility_data()