import pandas as pd
import numpy as np
import ast

from shared_components.model_data import ModelSuperRunData
from shared_components.logger import log
from .chromosome import Chromosome


class DataAnalysis:
    def __init__(
        self, data: list[ModelSuperRunData] = None, data_df: pd.DataFrame = None
    ):
        if data is not None:
            self.data = data
            self.raw_data_df = self.data_to_df(data)
        elif data_df is not None:
            self.raw_data_df = self.parse_input_data_df(data_df)

    def data_to_df(self, data) -> pd.DataFrame:
        df = pd.concat([model_run_data.to_dataframe() for model_run_data in data])
        print(f"df memory usage: {df.memory_usage().sum() / 1024 ** 2} MB")
        return df

    def parse_input_data_df(self, df) -> pd.DataFrame:
        # parse strings back to dicts and lists
        df["average_fitness_per_species"] = df["average_fitness_per_species"].apply(
            lambda x: ast.literal_eval(x)
        )
        df["best_fitness_per_species"] = df["best_fitness_per_species"].apply(
            lambda x: ast.literal_eval(x)
        )
        df["chromosome"] = df["chromosome"].apply(lambda x: ast.literal_eval(x))
        return df

    def analyze_data(self):
        # result = self.raw_data_df.groupby("parameter_id")["predator_kills"].mean()
        # print(result)
        # # self.analysis_df[]

        # No data analysis yet.
        pass

    def data_to_csv(self, filepath="output.csv"):
        self.raw_data_df.to_csv(filepath, index=False)

    def last_n_rows_to_csv(self, n, filepath="output.csv"):
        self.raw_data_df.tail(n).to_csv(filepath, index=False)

    def generate_report(self, verbosity=1) -> str:
        if verbosity == 1:
            return self.generate_report_type_1()
        elif verbosity == 2:
            return self.generate_report_type_2()
        else:
            raise ValueError("Invalid verbosity level.")

    def generate_report_type_1(self) -> str:
        # Keep only one row per generation & super run, the benchmark score, and the best and average fitness.
        # -> the data is the same in all columns for the same generation, so we just need one row per generation (per run).
        df = self.raw_data_df.dropna(subset=["benchmark_score"])

        df = df.drop_duplicates(
            subset=["super_run_number", "generation_number"], keep="last", inplace=False
        )
        df = df[
            [
                "generation_number",
                "benchmark_score",
                "best_fitness_per_species",
                "average_fitness_per_species",
            ]
        ]
        df = df.groupby(["generation_number"]).mean()
        report = df.to_string()
        report += (
            f"\n\nBest chromosomes: \n{self.get_best_chromosome_set().to_string()}"
        )
        return report

    def generate_report_type_2(self) -> str:
        df = self.get_report_type_2_df()
        report = df.to_string()
        report += f"\n\nChromosome combinations: \n{self.get_chromosome_combinations_report_df().to_string()}"
        report += (
            f"\n\nBest chromosomes: \n{self.get_best_chromosome_set().to_string()}"
        )
        return report

    def generate_report_type_3(self) -> str:
        return "Empty report, function verbosity level 3 not implemented yet."

    def get_report_type_2_df(self) -> pd.DataFrame:
        # Keep only one row per generation, the benchmark score, and the best and average fitness.
        df = self.raw_data_df.dropna(subset=["benchmark_score"])
        df = df.drop_duplicates(
            subset=["super_run_number", "generation_number"], keep="last", inplace=False
        )
        df["average_fitness_of_all"] = df["average_fitness_per_species"].apply(
            lambda x: sum(x.values()) / len(x.values())
        )
        df = df[
            [
                "generation_number",
                "benchmark_score",
                "average_fitness_of_all",
                "best_fitness_per_species",
                "average_fitness_per_species",
            ]
        ]
        df = df.groupby(["generation_number"]).mean(numeric_only=True).reset_index()
        # df = df.loc[df["generation_number"] != 0]
        return df

    def get_chromosome_combinations_report_df(self) -> pd.DataFrame:
        df = self.raw_data_df.dropna(subset=["benchmark_score"])
        df = df.drop_duplicates(
            subset=["super_run_number", "generation_number"], keep="last", inplace=False
        )
        df = df[
            [
                "super_run_number",
                "generation_number",
                "benchmark_best_combination",
                "benchmark_best_combination_score",
                "benchmark_best_combination_count",
                "benchmark_worst_combination",
                "benchmark_worst_combination_score",
            ]
        ]
        species_count_df = self.get_species_count_df()

        df = pd.merge(
            df, species_count_df, on=["super_run_number", "generation_number"]
        )
        return df

    def get_best_chromosome_set(self) -> pd.DataFrame:
        df = self.raw_data_df.loc[
            self.raw_data_df["benchmark_score"]
            == self.raw_data_df["benchmark_score"].max()
        ]
        df = df.loc[df["generation_number"] == df["generation_number"].max()]
        df = df.loc[df["total_run_number"] == df["total_run_number"].max()]

        df = df[
            [
                "benchmark_score",
                "super_run_number",
                "generation_number",
                "individual_fitness",
                "collective_fitness",
                "chromosome",
                "chromosome_type",
                "chromosome_id",
            ]
        ]

        return df

    def get_species_count_df(self):
        # Get the data we need
        df = self.raw_data_df[
            [
                "super_run_number",
                "generation_number",
                "chromosome_id",
                "chromosome_type",
            ]
        ]
        # Apply filter
        df = df.drop_duplicates(
            subset=["super_run_number", "generation_number", "chromosome_id"],
            keep="last",
            inplace=False,
        )
        # Calculate the information we want
        df = df.groupby(["super_run_number", "generation_number", "chromosome_type"])[
            "chromosome_id"
        ].count()
        # Format the data
        df = df.reset_index()
        df = df.pivot(
            index=["super_run_number", "generation_number"],
            columns="chromosome_type",
            values="chromosome_id",
        )
        df = df.fillna(0)
        df.columns = df.columns.map(lambda name: f"species_{name}")
        df = df.reset_index()
        return df

    def split_dict_column_to_columns(self, df, column_name) -> pd.DataFrame:
        dicts = df[column_name].to_list()
        index = df.index
        split_df = pd.DataFrame(dicts, index=index)
        split_df = split_df.add_prefix(column_name)
        return split_df

    def generate_summary(self) -> pd.DataFrame:
        df = self.raw_data_df.dropna(subset=["benchmark_score"])
        df = df.drop_duplicates(
            subset=["super_run_number", "generation_number"], keep="last"
        )
        df["average_fitness_of_all"] = df["average_fitness_per_species"].apply(
            lambda x: sum(x.values()) / len(x.values())
        )
        df = pd.merge(
            df[
                [
                    "super_run_number",
                    "generation_number",
                    "benchmark_score",
                    "benchmark_best_combination_score",
                    "average_fitness_of_all",
                ]
            ],
            self.get_species_count_df(),
            on=["super_run_number", "generation_number"],
        )
        return df

    def get_chromosomes_for_generation_and_run(
        self, generation_number, super_run_number
    ) -> pd.DataFrame:
        df = self.raw_data_df.loc[
            (self.raw_data_df["generation_number"] == generation_number)
            & (self.raw_data_df["super_run_number"] == super_run_number)
        ]
        df = df.drop_duplicates(subset=["chromosome_id"], keep="first", inplace=False)
        return df

    # Get the data on just one subject. Easier to build the plots.
    def get_benchmark_data(self) -> pd.DataFrame:
        df = self.raw_data_df.dropna(subset=["benchmark_score"])
        df = df.drop_duplicates(
            subset=["super_run_number", "generation_number"], keep="last", inplace=False
        )
        df = df[
            [
                "generation_number",
                "benchmark_score",
                "benchmark_best_combination_score",
                "benchmark_best_combination_count",
                "benchmark_worst_combination_score",
            ]
        ]
        df = df.groupby(["generation_number"]).mean().reset_index()
        return df

    def get_fitness_data(self) -> pd.DataFrame:
        # Probably needs some looking into -- individual and collective fitness average to the same TODO
        df = self.raw_data_df[
            [
                "super_run_number",
                "generation_number",
                "individual_fitness",
                "collective_fitness",
                "combined_fitness",
            ]
        ]

        # First, calculate per run
        metrics = ["individual_fitness", "collective_fitness", "combined_fitness"]
        result_dfs = []

        for metric in metrics:
            result = df.groupby(["super_run_number", "generation_number"])[metric].agg(
                ["mean", "max", "min"]
            )
            result.columns = [f"{stat}_{metric}" for stat in result.columns]
            result_dfs.append(result)

        result_df = pd.concat(result_dfs, axis=1)
        result_df = result_df.sort_values(
            ["super_run_number", "generation_number"]
        ).reset_index()

        # Then, calculate the average across runs
        result_df = result_df.groupby(["generation_number"]).mean().reset_index()
        result_df = result_df.drop(["super_run_number"], axis=1)

        return result_df

    def get_species_data(self, sort_species=True) -> pd.DataFrame:
        df = self.get_species_count_df()
        df = df.groupby("generation_number").mean()
        df = df.reset_index()
        log.warning(
            "Creating species data average. If used to make a plot, it will be wrong."
        )
        return df

    def sort_species_by_value(self, species_df) -> pd.DataFrame:
        # Get the names of the rank columns
        rank_cols = [col for col in species_df.columns if col.startswith("species_")]

        # Sort the values in each row
        for i, row in species_df.iterrows():
            values = row[rank_cols].values
            sorted_indices = np.argsort(-values)
            sorted_values = values[sorted_indices]
            for j, col in enumerate(rank_cols):
                species_df.at[i, col] = sorted_values[j]

        species_df.rename(
            columns={col: f"species_{j+1}" for j, col in enumerate(rank_cols)},
            inplace=True,
        )
        return species_df

    def get_chromosome_data(self) -> pd.DataFrame:
        pass

    def get_chromosomes_for_generation_and_run_as_chromosome(
        self, generation_number, super_run_number
    ) -> list[Chromosome]:
        df = self.get_chromosomes_for_generation_and_run(
            generation_number, super_run_number
        )
        chromosomes = []
        for _, row in df.iterrows():
            chromosomes.append(
                Chromosome(
                    genes=row["chromosome"],
                    chromosome_id=row["chromosome_id"],
                    chromosome_type=row["chromosome_type"],
                )
            )
        return chromosomes


if __name__ == "__main__":
    pass
