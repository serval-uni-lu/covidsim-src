import logging

import wget

from src.utils.config_loader import get_config
import os

LOGGER = logging.getLogger()
CONFIG = get_config()


def download(url, path):
    LOGGER.info(f"Downloading {url} -> {path}")
    if os.path.exists(path):
        os.remove(path)
    wget.download(url, path)


def process_url(urls, raw_path):

    if isinstance(urls, dict):
        for l_key in urls:
            download(urls[l_key], raw_path[l_key])
    else:
        download(urls, raw_path)


def run():
    logging.basicConfig()
    LOGGER.setLevel(logging.INFO)
    LOGGER.info("Starting...")
    datasets = CONFIG["common"]["datasets"]
    for d_key in datasets:
        if "url" in datasets[d_key]:
            urls = datasets[d_key]["url"]
            raw_path = datasets[d_key]["raw"]
            process_url(urls, raw_path)

    LOGGER.info("Done.")


if __name__ == "__main__":
    run()
