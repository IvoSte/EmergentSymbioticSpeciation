import os
from dynaconf import Dynaconf
from models.model_factory import ModelType
from shared_components.logger import log


def check_config(config):
    # Check for config settings that should go together
    # List is inexhaustive, some permutations may still not work

    if config["EVOLUTION_TYPE"] == "forced_subpopulations":
        assert (
            config["CHROMOSOME_SAMPLING"] == "deterministic"
            or config["CHROMOSOME_SAMPLING"] == "evolutionary"
            or config["CHROMOSOME_SAMPLING"] == "simple_random"
        ), "Forced subpopulations only works with deterministic sampling. Allowing evolutionary and random sampling for testing purposes."
        if (
            config["CHROMOSOME_SAMPLING"] == "evolutionary"
            or config["CHROMOSOME_SAMPLING"] == "simple_random"
        ):
            log.warning(
                f"Runninng forced subpopulations with {config['CHROMOSOME_SAMPLING']} sampling. This is not recommended."
            )
    if config["MODEL"] == "toxin":
        assert (
            config["N_TOXINS"] == config["CHROMOSOME_LENGTH"]
        ), "N_TOXINS must equal CHROMOSOME_LENGTH, genes determine the function per toxin in a 1:1 mapping"

    if config["MODEL"] == "predator_prey":
        assert (
            config["N_AGENTS"] == config["N_PREDATORS"]
        ), "N_AGENTS must equal N_PREDATORS"
    if (
        config["SUBPOPULATION_SIZE"] * config["N_SUBPOPULATIONS"]
        > config["N_AGENTS"] * config["N_COMPOSITIONS"]
    ):
        log.error(
            """\nPopulation is bottlenecked by the number of parameter sets / compositions."""
            """\nPopulation size (types * chromosomes per type) is greater than the number of slots per generation (agents * parameter sets / compositions)."""
            """\nThis will result in some chromosomes never being selected for reproduction, reducing the population size."""
        )
    if (
        config["MODEL"] == "function_optimization"
        and config["CHROMOSOME_SAMPLING"] == "deterministic"
    ):
        assert (
            config["N_AGENTS"] == config["N_SUBPOPULATIONS"]
        ), "N_AGENTS must equal to number of chromosome types for deterministic sampling"
    if (
        config["CHROMOSOME_SAMPLING"] == "evolutionary"
        and config["COMPOSITION"]["CHROMOSOME_RESIZING"]
        and config["N_SUBPOPULATIONS"] * config["SUBPOPULATION_SIZE"]
        < config["N_COMPOSITIONS"]
    ):
        log.warning(
            "Composition chromosome resizing is enabled and there are not enough compositions to slot all chromosomes if composition size becomes 1. This could result in an error aligning the population and compositions. Recommend increasing the number of compositions, decreasing the number of chromosomes or disabling composition chromosome resizing."
        )
    if config["CHROMOSOME_SAMPLING"] == "evolutionary" and config[
        "COMPOSITION.MUTATION_PROBABILITY"
    ] > (1 / config["N_AGENTS"]):
        log.warning(
            "Composition mutation rate too high. Aim for 1/chromosome lenght, which is 1/N_AGENTS for compositions."
        )
    # TODO Add any config restrictions or warnings here.


def load_config(model_type: ModelType = None, config_filepath: str = None):
    if config_filepath == None:
        if model_type == None:
            raise Exception("No config filepath or model type provided.")
        else:
            config_filepath = os.path.join("config", f"{model_type.value}_config.toml")

    if not os.path.exists(config_filepath):
        raise Exception("Config file not found")

    config = Dynaconf(settings_files=[config_filepath])
    check_config(config)
    return config
