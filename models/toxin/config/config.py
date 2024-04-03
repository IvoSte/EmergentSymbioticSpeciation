from dynaconf import Dynaconf
import os

configFilePath = os.path.join("config", "config.toml")

config = Dynaconf(settings_files=[configFilePath])
