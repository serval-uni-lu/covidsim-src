import logging
from utils.config_loader import get_config
from utils import load_dataset
LOGGER = logging.getLogger()
CONFIG = get_config()

def merge_results():
    oxford = _get_csv_as_dataframe(_data_root, 'oxford')
    gmobility = _get_csv_as_dataframe(_data_root, 'gmobility')
    demographic = _get_csv_as_dataframe(_data_root, 'demographic')
    rt_estimation = _get_csv_as_dataframe(_data_root, 'rt_estimation')
    country_metrics = _get_csv_as_dataframe(_data_root, 'country_metrics')

    out = pd.merge(oxford, gmobility, how="inner", on=["CountryName", "Date"])
    out = pd.merge(out, rt_estimation, how="inner", on=["CountryName", "Date"])
    out = pd.merge(out, demographic, how="inner", on=["CountryName"])
    out = pd.merge(out, country_metrics, how="inner", on=["CountryName"])

    merged_data_dir = os.path.join(_data_root, 'merged')

    if not os.path.exists(merged_data_dir):
        os.mkdir(merged_data_dir)

    out.to_csv(os.path.join(merged_data_dir, 'data.csv'), index=False)

def run():
    df = load_dataset()
    df = df.dropna()
    df.to_csv(CONFIG["common"]["paths"]["ml_features"])


if __name__ == "__main__":
    run()
