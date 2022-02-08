from src.utils import config_loader


def test_config_loading():
    config = config_loader.get_config("./config/config.yaml")
    assert config is not None
