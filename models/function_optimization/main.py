import sys

sys.path.insert(0, "../../")

from shared_components.logger import log
from config.config import config
from model.function_model import BenchmarkFunctionModel
from model.parameters import BenchmarkFunctionParameters


def minimal_run():
    log.info("Starting program -- minimal run")
    parameters = BenchmarkFunctionParameters.from_config(config)
    model = BenchmarkFunctionModel(parameters)
    model.run()
    print(model.get_data().to_dataframe())
    log.info("Model run successfully completed")


def run_model():
    minimal_run()


def main():
    run_model()


if __name__ == "__main__":
    main()
