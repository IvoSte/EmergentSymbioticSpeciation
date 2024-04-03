from enum import Enum
import sys
from dynaconf import Dynaconf, loaders
from dynaconf.utils.boxing import DynaBox
from itertools import product
from datetime import datetime
import pyfiglet
import os

sys.path.insert(0, "../")
from shared_components.logger import log
from shared_components.config_loader import check_config


class GeneratorMode(Enum):
    SINGLE_VALUE = "single_value"  # One config per setting change
    SWEEP = "sweep"  # One config per combination of settings
    BATCH = "batch"  # One config per setting change, but with multiple settings per config -> all first values, all second values, etc.


class NameMode(Enum):
    STANDARD = "standard"  # Standard name, e.g. 0.toml
    SHORT = "short"  # Short name, e.g. 0_mate_selection_type-random.toml
    LONG = "long"  # Long name, e.g. 0_mate_selection_type-random_reproduce_fraction-0.1.toml


class ExperimentGenerator:
    def __init__(
        self,
        base_config_path,
        overwrite_existing_experiments=False,
        name_mode: NameMode = NameMode.STANDARD,
        check_config_validity=True,
    ):
        assert os.path.exists(
            base_config_path
        ), f"Base config file not found: {base_config_path}. A base config file is required to generate experiments."

        self.base_config_path = base_config_path
        self.overwrite_existing_experiments = overwrite_existing_experiments
        if type(name_mode) == str:
            self.name_mode = NameMode(name_mode)
        self.name_mode = name_mode
        self.check_config_validity = check_config_validity

    def generate_experiments(
        self,
        experiment_path,
        settings,
        repetitions=1,
        generator_mode: GeneratorMode = GeneratorMode.SINGLE_VALUE,
    ):
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
            os.makedirs(os.path.join(experiment_path, "configs"))

        if generator_mode == GeneratorMode.SINGLE_VALUE:
            self.generate_experiments_set(experiment_path, settings, repetitions)
        elif generator_mode == GeneratorMode.SWEEP:
            self.generate_experiments_sweep(experiment_path, settings, repetitions)
        elif generator_mode == GeneratorMode.BATCH:
            self.generate_experiments_batch(experiment_path, settings, repetitions)

    def generate_experiments_set(self, experiment_path, settings, repetitions=1):
        for setting, values in settings.items():
            configs = self.generate_configs_from_list(setting, values)
            for config in configs:
                for i in range(repetitions):
                    filename = self.generate_config_name(
                        setting={setting: config[setting]}, prefix=i
                    )
                    filepath = os.path.join(experiment_path, "configs", filename)
                    self.save_config(filepath, config)

    def generate_experiments_sweep(self, experiment_path, settings, repetitions=1):
        if len(settings) >= 3:
            log.warning(
                "Sweep mode is not recommended for more than 2 settings. File names can get very long, and many experiments are produced."
            )
        setting_combinations = self.generate_settings_combinations(settings)
        configs = self.generate_configs_from_combination_list(setting_combinations)
        for config, settings in zip(configs, setting_combinations):
            for i in range(repetitions):
                filename = self.generate_config_name(setting=settings, prefix=i)
                filepath = os.path.join(experiment_path, "configs", filename)
                self.save_config(filepath, config)

    def generate_experiments_batch(self, experiment_path, settings, repetitions=1):
        # Check if all lists of settings are the same length
        if not all(
            len(v) == len(next(iter(settings.values()))) for v in settings.values()
        ):
            log.error("All lists of settings must be the same length in batch mode.")
            return

        settings_batches = self.generate_settings_batch(settings)
        configs = self.generate_configs_from_combination_list(settings_batches)
        for config, settings in zip(configs, settings_batches):
            for i in range(repetitions):
                filename = self.generate_config_name(setting=settings, prefix=i)
                filepath = os.path.join(experiment_path, "configs", filename)
                self.save_config(filepath, config)

    def generate_settings_combinations(self, settings):
        setting, values = zip(*settings.items())
        combinations = product(*values)
        setting_combinations = [
            dict(zip(setting, combination)) for combination in combinations
        ]
        return setting_combinations

    def generate_settings_batch(self, settings):
        settings_batches = []
        for values in zip(*settings.values()):
            batch = {key: value for key, value in zip(settings.keys(), values)}
            settings_batches.append(batch)
        return settings_batches

    def generate_config_name(self, setting, config=None, prefix=0):
        if self.name_mode == NameMode.STANDARD:
            return self.generate_standard_config_name(setting, config, prefix)
        elif self.name_mode == NameMode.SHORT:
            return self.generate_short_config_name(setting, prefix)
        elif self.name_mode == NameMode.LONG:
            return self.generate_long_config_name(setting, prefix)
        else:
            log.error(f"Name mode not recognized: {self.name_mode}")
            exit(1)

    def generate_standard_config_name(self, setting, config, prefix=0):
        value = config[setting]
        value = str(value).replace(".", "_")
        prefix = f"{prefix}_"
        return f"{prefix}{setting.lower()}_{value}.toml"

    def generate_long_config_name(self, settings, prefix):
        name = f"{prefix}"
        for k, v in settings.items():
            v = str(v).replace(".", "_")
            name += f"_{k.lower()}_{v}"
        name += ".toml"
        return name

    def generate_short_config_name(self, settings, prefix):
        name = f"{prefix}"
        for k, v in settings.items():
            k = "".join([c[0] for c in str(k).replace(".", "_").split("_")])
            v = str(v).replace(".", "_")
            v = v if v[0].isdigit() else "".join([c[0] for c in v.split("_")])
            name += f"_{k.lower()}-{v}"
        name += ".toml"
        return name

    def generate_configs_from_list(self, setting, values):
        configs = []
        for value in values:
            configs.append(self.generate_config(setting, value))
        return configs

    def generate_configs_from_combination_list(self, setting_combinations: list[dict]):
        configs = []
        for setting_combination in setting_combinations:
            configs.append(self.generate_config_from_setting(setting_combination))
        return configs

    def check_config(self, config):
        if self.check_config_validity:
            check_config(config)

    def generate_config(self, setting, value):
        config = self.load_config(self.base_config_path)
        config = self.set_config(config, setting, value)
        self.check_config(config)
        return config

    def set_config(self, config, setting, value):
        subconfig = config
        table_names = setting.split(".")[:-1]
        setting = setting.split(".")[-1]
        for table in table_names:
            if table not in subconfig:
                log.error(f"Table '{table}' not found in config")
                exit(1)
            subconfig = subconfig[table]
        if setting not in subconfig:
            log.error(f"Setting '{setting}' not found in config")
            exit(1)
        subconfig[setting] = value
        return config

    def generate_config_from_setting(self, settings):
        config = self.load_config(self.base_config_path)
        for setting, value in settings.items():
            config = self.set_config(config, setting, value)
        self.check_config(config)
        return config

    def create_note(self, experiment_path, note_text):
        with open(os.path.join(experiment_path, "note.txt"), "w") as f:
            f.write(f"EXPERIMENT: {experiment_path}\n")
            f.write(f"DATE CREATED: {datetime.now()}\n")
            f.write(f"NOTE: {note_text}\n")

    def load_config(self, filepath):
        return Dynaconf(settings_files=[filepath])

    def save_config(self, filepath, config):
        if os.path.exists(filepath):
            if self.overwrite_existing_experiments:
                log.warning(f"Overwriting existing experiment config: {filepath}")
            else:
                log.error(f"Experiment config already exists: {filepath}")
                return

        loaders.write(filepath, DynaBox(config.as_dict()).to_dict())
        log.info(f"Experiment {filepath} generated.")


def main():
    # Change these settings to generate different experiments -- metadata
    model = "predator_prey"  # options: toxin, predator_prey, function_optimization
    base_config = "testing_base_config"  # The base config, any parameter changes will be applied to this config. The base config must exist in the base_configs folder. Another config can be used as a base config.
    experiment_name = "your_experiment_name"  # The name of the experiment

    experiment_note = (
        # "example note -- describe the intent of the experiment here. Bonus for expected results. Keep it short."
        """
        Objective: _objective of the experiment_
        Result: _expected results, metrics to look at after running_
        Variables: _variables that change between experiments_
        Note: _any additional notes, specifically on the intent or naming_
        """
    )

    # Don't change these settings /{ #####
    experiment_path = os.path.join("experiments", model, experiment_name)
    base_config_path = os.path.join(
        "experiments", model, "base_configs", f"{base_config}.toml"
    )
    experiment_generator = ExperimentGenerator(
        base_config_path=base_config_path,
        overwrite_existing_experiments=True,
        name_mode=NameMode.SHORT,
    )
    # }/ #####

    # Change these settings to generate different experiments -- config options
    # In the settings argument, the key is the setting to change in the config, and the value is a list of values to try
    # The number of repetitions is how often each config is repeated.
    # the generator mode determines how the settings are combined and how many experiments are generated. The three options are:
    # - SINGLE_VALUE: One config per setting change
    # - SWEEP: One config per combination of settings
    # - BATCH: One config per setting change, but with multiple settings per config -> all first values, all second values, etc.
    experiment_generator.generate_experiments(
        experiment_path=experiment_path,
        settings={
            "N_RUNS": [10],
            "N_GENERATIONS": [100],
            "MATE_SELECTION_TYPE": ["random", "nearest_neighbour_from_sample"],
            "SCALE_FITNESS_WITH_REPRESENTATION": [True, False],
        },
        repetitions=4,
        generator_mode=GeneratorMode.SWEEP,
    )

    experiment_generator.create_note(experiment_path, experiment_note)


if __name__ == "__main__":
    ascii_banner = pyfiglet.figlet_format("Generating Experiments...")
    print(ascii_banner)
    main()
