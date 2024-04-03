from datetime import datetime
import subprocess
from models.model_factory import ModelFactory, ModelType
from shared_components.model_data import ModelSuperRunData
from speciation.data_analysis import DataAnalysis
from speciation.model_plot_generator import ModelPlotGenerator
from visualization import Visualizer
from shared_components.logger import log
import os
from dynaconf import Dynaconf


class OutputManager:
    def __init__(
        self,
        model_name,
        experiment_foldername,
        experiment_set_name,
        overwrite_existing_reports=False,
        make_reports=True,
        make_summary=True,
        make_plots=True,
        make_full_csv=True,
        make_model_plot=False,
        model_plot_separate_plots=False,
        visualize_chromosomes=True,
        report_verbosity_level=2,
        plot_verbosity_level=2,
        chromosome_visualization_techniques=["heatmap", "t-sne", "umap", "pca"],
        plots_to_generate=["benchmark", "fitness", "species"],
    ):
        self.model_name = model_name
        self.experiment_output_filepath = os.path.join(
            experiment_foldername, experiment_set_name
        )
        self.experiment_set_name = experiment_set_name
        self.overwrite_existing_reports = overwrite_existing_reports

        self.chromosome_visualization_techniques = chromosome_visualization_techniques
        self.plots_to_generate = plots_to_generate

        self.make_reports = make_reports
        self.make_summary = make_summary
        self.make_plots = make_plots
        self.make_model_plot = make_model_plot
        self.model_plot_separate_plots = model_plot_separate_plots
        self.make_full_csv = make_full_csv
        self.visualize_chromosomes = visualize_chromosomes

        self.report_verbosity_level = report_verbosity_level
        self.plot_verbosity_level = plot_verbosity_level

        self.create_output_folders()

    def create_output_folders(self):
        log.info(f"Creating experiment set '{self.experiment_set_name}'.")
        folder_names = [
            "reports",
            "summary",
            "csv",
            "plots",
            "chromosome_visualization",
            "model_plot",
        ]
        self.output_paths = {
            name: os.path.join(self.experiment_output_filepath, name)
            for name in folder_names
        }

        for folder_name in folder_names:
            if not os.path.exists(self.output_paths[folder_name]):
                if folder_name == "reports" and not self.make_reports:
                    continue
                if folder_name == "summary" and not self.make_summary:
                    continue
                if folder_name == "csv" and not self.make_full_csv:
                    continue
                if folder_name == "plots" and not self.make_plots:
                    continue
                if (
                    folder_name == "chromosome_visualization"
                    and not self.visualize_chromosomes
                ):
                    continue
                if folder_name == "model_plot" and not self.make_model_plot:
                    continue
                os.mkdir(self.output_paths[folder_name])
            elif folder_name == "reports" and self.overwrite_existing_reports:
                log.info(
                    f"Experiment set '{self.experiment_set_name}' already exists. Overwriting existing reports."
                )

    def generate_experiment_output(
        self,
        data: list[ModelSuperRunData],
        experiment_name: str,
        experiment_config: Dynaconf,
        visualize_chromosome_generations: list[int] = [],
    ):
        data_analysis = DataAnalysis(data)

        # Generate the reports: textual overview of the most salient results
        if self.make_reports:
            self.generate_report(
                data_analysis, experiment_name, self.report_verbosity_level
            )

        # Generate the summary: a csv file that can be averaged over multiple experiments
        if self.make_summary:
            self.generate_summary(data_analysis, experiment_name)

        # Generate the full csv: a csv file that contains all the data
        if self.make_full_csv:
            self.generate_full_csv(data_analysis, experiment_name)

        # Generate the plots: graphs that provide a visual representation of the results
        if self.make_plots:
            self.generate_plots(
                data_analysis,
                experiment_name,
                self.plots_to_generate,
                self.plot_verbosity_level,
            )

        if self.make_model_plot:
            self.generate_model_plot(data, experiment_name, experiment_config)

        if self.visualize_chromosomes:
            self.generate_chromosome_visualization(
                data_analysis,
                experiment_name,
                self.chromosome_visualization_techniques,
                visualize_chromosome_generations,
            )

    def generate_report(
        self,
        data_analysis: DataAnalysis,
        experiment_name: str,
        verbosity_level: int = 0,
    ):
        report = data_analysis.generate_report(verbosity_level)
        report_filepath = os.path.join(
            self.output_paths["reports"], f"{experiment_name}.txt"
        )
        self.save_report(report_filepath, report)

    def save_report(self, filepath: str, report: str):
        writing_mode = "w" if self.overwrite_existing_reports else "a"
        with open(filepath, writing_mode) as f:
            f.write(report)
            f.write("\n========================================\n")

    def generate_summary(self, data_analysis: DataAnalysis, experiment_name: str):
        summary = data_analysis.generate_summary()
        summary_filepath = os.path.join(
            self.output_paths["summary"], f"{experiment_name}_summary.csv"
        )
        summary.to_csv(summary_filepath, index=False)

    def generate_full_csv(self, data_analysis: DataAnalysis, experiment_name: str):
        csv_filepath = os.path.join(self.output_paths["csv"], f"{experiment_name}.csv")
        data_analysis.data_to_csv(csv_filepath)

    def generate_plots(
        self,
        da: DataAnalysis,
        experiment_name: str,
        plots_to_generate: list[str] = [],
        verbosity_level: int = 2,
    ):
        v = Visualizer()
        plots_filepath = os.path.join(self.output_paths["plots"], f"{experiment_name}")
        if "benchmark" in plots_to_generate:
            v.generate_plot(
                da.get_benchmark_data(),
                plot_type="benchmark",
                display=False,
                save=True,
                path=plots_filepath,
                index=None,
            )
        if "fitness" in plots_to_generate:
            v.generate_plot(
                da.get_fitness_data(),
                plot_type="fitness",
                display=False,
                save=True,
                path=plots_filepath,
                index=None,
            )
        if "species" in plots_to_generate:
            v.generate_plot(
                da.get_species_data(),
                plot_type="species",
                display=False,
                save=True,
                path=plots_filepath,
                index=None,
            )

    def generate_chromosome_visualization(
        self,
        da: DataAnalysis,
        experiment_name: str,
        techniques: list[str],
        generations: list[int],
        super_run: int = 0,
    ):
        v = Visualizer()
        chromosome_visualization_filepath = os.path.join(
            self.output_paths["chromosome_visualization"], f"{experiment_name}"
        )
        for technique in techniques:
            for generation in generations:
                v.visualize_chromosomes(
                    da.get_chromosomes_for_generation_and_run_as_chromosome(
                        generation_number=generation, super_run_number=super_run
                    ),
                    technique,
                    display=False,
                    save=True,
                    path=os.path.join(chromosome_visualization_filepath, technique),
                    index=generation,
                )

    def generate_model_plot(self, data, experiment_name, experiment_config):
        model_visualizer = ModelPlotGenerator(experiment_config)
        v = model_visualizer.visualize_results(data, self.model_plot_separate_plots)
        v.save(
            filepath=os.path.join(self.output_paths["model_plot"], f"{experiment_name}")
        )

    def update_note(self, elapsed_time):
        with open(os.path.join(self.experiment_output_filepath, "note.txt"), "r+") as f:
            lines = f.readlines()
            if lines[-3].startswith("DATE EXECUTED:"):
                lines = lines[:-3]
            lines.append(f"DATE EXECUTED: {datetime.now()}\n")
            lines.append(
                f"EXECUTION TIME: {elapsed_time//3600} hour(s) {(elapsed_time%3600)//60} minute(s) {round(elapsed_time%60,2)} second(s).\n"
            )
            lines.append(f"MODEL GIT VERSION: {self.get_model_version()}\n")
            f.seek(0)
            f.writelines(lines)
            f.truncate()

    def get_model_version(self):
        commit_version = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode("utf-8")
        )
        return commit_version
