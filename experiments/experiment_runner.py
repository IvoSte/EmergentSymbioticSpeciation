import gc
import psutil
import sys

sys.path.insert(0, "../")
import os
import time
from shared_components.logger import log
import glob
from dynaconf import Dynaconf
from shared_components.model_data import ModelSuperRunData
from speciation.model_runner import ModelRunner
from shared_components import config_loader
from shared_components.threadpool import MultiProcessor
from output_manager import OutputManager


class ExperimentRunner:
    def __init__(
        self,
        model_name,
        experiment_set_name,
        experiment_foldername="experiments",
        run_existing_experiments_again=False,
        overwrite_existing_reports=False,
        multi_processing=True,
        max_num_cpu=None,
        generate_reports=True,
        generate_plots=True,
        generate_model_plot=False,
        model_plot_separate_plots=False,
        visualize_chromosomes=True,
        generate_summary_csv=True,
        save_full_csv=False,
        report_verbosity_level=2,
        plot_verbosity_level=2,
        chromosome_visualization_techniques=["heatmap", "t-sne", "umap", "pca"],
        plots_to_generate=["benchmark", "fitness", "species"],
    ):
        # Names
        self.experiment_foldername = experiment_foldername
        self.experiment_set_name = experiment_set_name
        self.experiment_config_filepath = os.path.join(
            self.experiment_foldername, f"{self.experiment_set_name}", "configs"
        )

        self.check_experiment_folder()

        self.output_manager = OutputManager(
            model_name=model_name,
            experiment_foldername=experiment_foldername,
            experiment_set_name=experiment_set_name,
            overwrite_existing_reports=overwrite_existing_reports,
            make_reports=generate_reports,
            make_summary=generate_summary_csv,
            make_full_csv=save_full_csv,
            make_plots=generate_plots,
            make_model_plot=generate_model_plot,
            model_plot_separate_plots=model_plot_separate_plots,
            visualize_chromosomes=visualize_chromosomes,
            report_verbosity_level=report_verbosity_level,
            plot_verbosity_level=plot_verbosity_level,
            chromosome_visualization_techniques=chromosome_visualization_techniques,
            plots_to_generate=plots_to_generate,
        )

        # Settings
        self.overwrite_existing_reports = overwrite_existing_reports
        self.run_existing_experiments_again = run_existing_experiments_again
        self.report_verbosity_level = report_verbosity_level
        self.multi_processing = multi_processing
        self.max_num_cpu = max_num_cpu

    def check_experiment_folder(self):
        assert os.path.exists(
            self.experiment_config_filepath
        ), f"No folder named '{self.experiment_config_filepath}' found. No experiments to run."

    def report_exists(self, report_filepath):
        return os.path.exists(report_filepath)

    def start(self):
        start = time.perf_counter()
        log.info(
            f"Starting experiment runner for experiment set: {self.experiment_set_name}",
        )

        if self.multi_processing:
            log.info(f"Using multi processing.")
            multiprocessor = MultiProcessor(self.max_num_cpu)
            # for config_name, config_filepath in self.get_next_config_filepath():
            #     multiprocessor.add_task(
            #         self.perform_experiment,
            #         config_name=config_name,
            #         config_filepath=config_filepath,
            #     )
            # multiprocessor.wait_completion()
            multiprocessor.run_pool(
                self.perform_experiment_unwrap, self.get_all_config_filespaths()
            )

        else:
            log.info(f"Using single processing. Are you sure?")
            for config_name, config_filepath in self.get_next_config_filepath():
                self.perform_experiment(config_name, config_filepath)

        finished = time.perf_counter()
        elapsed_time = round(finished - start, 2)
        print(
            f"Finished in {elapsed_time//3600} hour(s) {(elapsed_time%3600)//60} minute(s) {round(elapsed_time%60,2)} second(s)."
        )
        process = psutil.Process(os.getpid())
        log.info(
            f"Memory usage point 5 - all processes completed: {process.memory_info().rss / 1024 ** 2} MB"
        )

        self.update_note(elapsed_time=elapsed_time)

    def get_next_config_filepath(self):
        config_files = glob.glob(
            os.path.join(self.experiment_config_filepath, "*.toml")
        )
        for config_filepath in config_files:
            config_name = os.path.split(config_filepath)[-1].split(".")[0]
            log.info(f"Loading config: {config_name}")
            yield (config_name, config_filepath)

    def get_all_config_filespaths(self):
        config_name_and_filepath = []
        for config_name, config_filepath in self.get_next_config_filepath():
            config_name_and_filepath.append((config_name, config_filepath))
        return config_name_and_filepath

    def load_config(self, config_filepath) -> Dynaconf:
        # Run through the config loader so it makes sure the config is valid
        try:
            config = config_loader.load_config(config_filepath=config_filepath)
        except AssertionError as e:
            log.error(
                f"Loading experiment config '{config_filepath}' failed with assertion error '{e}'"
            )
        return config

    def perform_experiment_unwrap(self, args):
        return self.perform_experiment(*args)

    def perform_experiment(self, config_name, config_filepath):
        if self.experiment_exists_already(config_name):
            if self.run_existing_experiments_again:
                log.info(f"Experiment '{config_name}' already exists. Running again.")
            else:
                log.info(f"Experiment '{config_name}' already exists. Skipping.")
                return

        experiment_start_time = time.perf_counter()
        log.info(f"Running experiment: {config_name}")

        config = self.load_config(config_filepath)

        try:
            data = self.run_experiment(config)
        except Exception as e:
            log.error(f"Experiment '{config_name}' failed with error: {e}")
            return
        process = psutil.Process(os.getpid())
        log.info(
            f"Memory usage point 1 - post running pre process: {process.memory_info().rss / 1024 ** 2} MB"
        )
        gc.collect()  # Free up memory. If it doesn't help, remove for speed. NOTE
        log.info(
            f"Memory usage point 2 - post clear 1: {process.memory_info().rss / 1024 ** 2} MB"
        )
        log.info(f"Experiment '{config_name}' completed. Generating output...")
        self.generate_output(data, config_name, config)
        log.info(
            f"Memory usage point 3 - post output generation: {process.memory_info().rss / 1024 ** 2} MB"
        )
        gc.collect()  # Free up memory. If it doesn't help, remove for speed. NOTE
        log.info(
            f"Memory usage point 4 - post clear 2: {process.memory_info().rss / 1024 ** 2} MB"
        )

        experiment_end_time = time.perf_counter()
        elapsed_time = round(experiment_end_time - experiment_start_time, 2)
        log.info(
            f"Experiment '{config_name}' complete in {elapsed_time//3600} hour(s) {(elapsed_time%3600)//60} minute(s) {round(elapsed_time%60,2)} second(s)."
        )

    def run_experiment(self, config: Dynaconf) -> list[ModelSuperRunData]:
        model_runner = ModelRunner(config)
        data = model_runner.run()
        return data

    def experiment_exists_already(self, config_name):
        return os.path.exists(
            os.path.join(
                self.output_manager.output_paths["reports"], f"{config_name}.txt"
            )
        )

    def generate_output(
        self, data: list[ModelSuperRunData], config_name: str, config: Dynaconf
    ):
        visualize_chromosome_generations = []
        if config["VISUALIZE_CHROMOSOMES"]:
            visualize_chromosome_generations = list(
                range(
                    0,
                    config["N_GENERATIONS"],
                    config["VISUALIZE_CHROMOSOMES_INTERVAL"],
                )
            )
        self.output_manager.generate_experiment_output(
            data=data,
            experiment_name=config_name,
            experiment_config=config,
            visualize_chromosome_generations=visualize_chromosome_generations,
        )

    def update_note(self, elapsed_time):
        self.output_manager.update_note(elapsed_time=elapsed_time)


def handle_commandline_arguments():
    parser = argparse.ArgumentParser(
        description="Run an experiment set and generate reports."
    )
    parser.add_argument(
        "--model_name",
        "-m",
        type=str,
        help="The model to run.",
        required=True,
    )
    parser.add_argument(
        "--experiment-set-name",
        "-e",
        type=str,
        help="The name of the experiment set to run.",
        required=True,
    )
    parser.add_argument(
        "--overwrite-existing-reports",
        "-o",
        action="store_true",
        help="Whether to overwrite existing reports.",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--run-existing-experiments-again",
        "-r",
        action="store_true",
        help="Whether to run existing experiments again.",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--multi-processing",
        "-mp",
        action="store_true",
        help="Whether to use multi processing.",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--max-num-cpus",
        "-cpu",
        type=int,
        help="The maximum number of CPUs to use. If not specified, all available CPUs will be used.",
        required=False,
        default=None,
    )
    args = parser.parse_args()
    return args


# @profile
def main():
    from postprocessing import PostProcessing

    # Example command line run:
    # python experiment_runner.py -m "predator_prey" -e "test_experiment" -o True -r True -mp True -cpu 4

    args = handle_commandline_arguments()
    experiment_config = Dynaconf(
        settings_files=[os.path.join("config", "experiment_runner_config.toml")]
    )
    experiment_config = experiment_config[args.model_name]

    experiment_runner = ExperimentRunner(
        model_name=args.model_name,
        experiment_set_name=args.experiment_set_name,
        experiment_foldername=os.path.join("experiments", args.model_name),
        overwrite_existing_reports=args.overwrite_existing_reports,
        run_existing_experiments_again=args.run_existing_experiments_again,
        multi_processing=args.multi_processing,
        max_num_cpu=args.max_num_cpus,
        generate_reports=experiment_config["GENERATE_REPORTS"],
        generate_plots=experiment_config["GENERATE_PLOTS"],
        generate_model_plot=experiment_config["GENERATE_MODEL_PLOT"],
        model_plot_separate_plots=experiment_config["MODEL_PLOT_SEPARATE_IMAGES"],
        visualize_chromosomes=experiment_config["VISUALIZE_CHROMOSOMES"],
        generate_summary_csv=experiment_config["GENERATE_SUMMARY_CSV"],
        save_full_csv=experiment_config["SAVE_FULL_CSV"],
        report_verbosity_level=experiment_config["REPORT_VERBOSITY_LEVEL"],
        plot_verbosity_level=experiment_config["PLOT_VERBOSITY_LEVEL"],
        chromosome_visualization_techniques=experiment_config[
            "CHROMOSOME_VISUALIZATION_TECHNIQUES"
        ],
        plots_to_generate=experiment_config["PLOTS_TO_GENERATE"],
    )
    experiment_runner.start()

    log.info("Post processing disabled. Code disabled until further notice.")
    postprocessing = False
    if postprocessing:
        PostProcessing().aggregate_summaries(
            experiment_runner.output_manager.experiment_summary_filepath
        )
        try:
            PostProcessing().plot_aggregates(
                experiment_runner.output_manager.experiment_summary_filepath
            )
        except Exception as e:
            log.error(f"Error plotting aggregates: {e}")


if __name__ == "__main__":
    import argparse

    main()
