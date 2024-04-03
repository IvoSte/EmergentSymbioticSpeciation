import glob
import os
import pandas as pd
from shared_components.logger import log


class PostProcessing:
    def __init__(self):
        pass

    def aggregate_summaries(self, summaries_filepath):
        summary_filepaths = glob.glob(os.path.join(summaries_filepath, "*.csv"))
        experiments = {}

        # Group summary files by experiment name
        for summary_filepath in summary_filepaths:
            basename = os.path.basename(summary_filepath)
            # Skip summary files of experiments that were executed only once
            if not basename.split("_")[0].isdigit():
                log.warning(
                    "Summary file name start with '#_', or the experiment is executed only once and an aggregate is not needed. It is possible that this warning shows up if the experiment is run again and the error is triggered from the already produced aggregate files."
                )
                continue
            experiment_name = basename.split("_", 1)[1]
            experiment_name = experiment_name.split(".csv")[0]
            if experiment_name not in experiments:
                experiments[experiment_name] = []
            experiments[experiment_name].append(summary_filepath)

        # Aggregate each experiment
        for experiment_name, summary_files in experiments.items():
            aggregate_df = self.aggregate_summary(summary_files)
            aggregate_df.to_csv(
                os.path.join(summaries_filepath, f"{experiment_name}_aggregate.csv"),
                index=True,
            )

    def aggregate_summary(self, summaries_filepaths):
        dataframes = [pd.read_csv(summary_file) for summary_file in summaries_filepaths]
        concatenated_df = pd.concat(dataframes)
        aggregate_df = (
            concatenated_df.groupby("generation_number")
            .agg(["mean", "std"])
            .reset_index()
        )
        return aggregate_df

    def plot_aggregates(self, aggregates_filepath):
        from visualization import Visualizer

        v = Visualizer()
        aggregate_filepaths = glob.glob(
            os.path.join(aggregates_filepath, "*_aggregate.csv")
        )
        for aggregate_filepath in aggregate_filepaths:
            basename = os.path.basename(aggregate_filepath)
            experiment_name = (
                basename.split("_", 1)[1]
                if basename.split("_")[0].isdigit()
                else basename.split(".csv")[0]
            )
            plot_path = os.path.join(aggregates_filepath, experiment_name)
            aggregate_df = pd.read_csv(aggregate_filepath, header=[0, 1], index_col=0)
            v.generate_plot(
                aggregate_df,
                plot_type="fitness_aggregate",
                display=False,
                save=True,
                path=plot_path,
                index="",
            )
            v.generate_plot(
                aggregate_df,
                plot_type="benchmark_aggregate",
                display=False,
                save=True,
                path=plot_path,
                index="",
            )


if __name__ == "__main__":
    post_processing = PostProcessing()
    post_processing.summarize_summaries(
        os.path.join("experiments", "summary_test", "summary")
    )
