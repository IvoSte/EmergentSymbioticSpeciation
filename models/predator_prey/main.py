# Relative imports are hard. This is a hack to make it work. NOTE Refactor this when there is time.
import sys


sys.path.insert(0, "../../")

from speciation.chromosome import Chromosome
from speciation.chromosome_util import load_chromosome_set
from config.config import config
from shared_components.logger import log
from model.predator_prey import PredatorPrey
from model.parameters import PPParameters
from model.event_manager import EventManager
from model.controller import Controller
from model.viewer.viewer import Viewer

# What is the purpose of this file?
# We want to run the model from the top level folder to use all speciation architecture
# We want to run the model from this folder without the speciation architecture
# We use this to visualize the model behaviour.


def minimal_run():
    log.info("Starting program -- minimal run")
    parameters = PPParameters.from_config(config)
    model = PredatorPrey(parameters)
    model.run()
    log.info("Model run successfully completed")


def minimal_run_with_viewer():
    assert config["RUN_WITH_VIEWER"] == True, "Viewer must be enabled for viewer run"
    log.info("Starting program -- minimal run with viewer")
    event_manager = EventManager()
    parameters = PPParameters.from_config(config)
    model = PredatorPrey(
        parameters=parameters,
        event_manager=event_manager,
        run_with_viewer=True,
        frames_per_second=config["FRAMES_PER_SECOND"],
    )
    controller = Controller(event_manager, model)
    viewer = Viewer(event_manager, model)

    model.run()
    log.info("Model run successfully completed")


def view_benchmark(chromosomes: list[Chromosome] = None):
    log.info("Starting program -- view benchmark")
    event_manager = EventManager()
    parameters = PPParameters.from_config(config)

    if chromosomes:
        parameters.agent_chromosomes = chromosomes

    model = PredatorPrey(
        parameters=parameters,
        event_manager=event_manager,
        run_with_viewer=True,
        frames_per_second=config["FRAMES_PER_SECOND"],
    )
    controller = Controller(event_manager, model)
    viewer = Viewer(event_manager, model)

    print(f"Successes: {model.run_benchmark()}")
    log.info("Model run successfully completed")


def run_model():
    minimal_run()
    minimal_run_with_viewer()
    view_benchmark()


def handle_commandline_arguments():
    parser = argparse.ArgumentParser(
        description="Run the Predator-Prey model in standalone configuration."
    )
    parser.add_argument(
        "--viewer",
        "-v",
        type=bool,
        help="Run the model with the viewer enabled.",
        required=True,
        default=False,
    )
    parser.add_argument(
        "--benchmark",
        "-b",
        type=bool,
        help="Run the model in benchmark mode or regular mode.",
        required=True,
        default=False,
    )
    parser.add_argument(
        "--chromosome_set",
        "-c",
        type=str,
        help="Path to a chromosome set to use for benchmarking.",
        required=False,
        default=None,
    )
    args = parser.parse_args()
    return args


def main():
    arguments = handle_commandline_arguments()
    config["RUN_WITH_VIEWER"] = arguments.viewer

    chromosomes = None
    if arguments.chromosome_set:
        chromosomes = load_chromosome_set(arguments.chromosome_set)
    if arguments.benchmark:
        view_benchmark(chromosomes)
    else:
        run_model()


if __name__ == "__main__":
    import argparse

    main()
