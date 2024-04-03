# Relative imports are hard. This is a hack to make it work. NOTE Refactor this when there is time.
import sys

sys.path.insert(0, "../../")

from config.config import config
from shared_components.logger import log
from model.toxin_model import ToxinModel
from model.parameters import ToxinParameters


def minimal_run():
    log.info("Starting program -- minimal run")
    parameters = ToxinParameters.from_config(config)
    model = ToxinModel(parameters)
    model.run()
    log.info("Model run successfully completed")


def run_model():
    minimal_run()


def main():
    run_model()


if __name__ == "__main__":
    main()
