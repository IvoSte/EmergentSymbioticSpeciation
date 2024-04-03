from math import ceil
import numpy as np
import pandas as pd
from .model_data import ModelSuperRunData
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP


class ModelAnalysis:
    def __init__(self, data: list[ModelSuperRunData], parameters):
        self.data = data
        self.raw_data_df = self.data_to_df(data)
        # self.data_df = self.preprocess_data()
        self.parameters = parameters

    def data_to_df(self, data):
        df = pd.concat(
            [model_super_run_data.to_dataframe() for model_super_run_data in data]
        )
        return df

    def sort_df_colums(self, df):
        sorted_columns = df.sum().sort_values(ascending=False).index
        return df[sorted_columns]

    def preprocess_data(self):
        # If we want to do any preprocessing of the data, we can do it here
        # You will need to replace raw_data_df with data_df in the functions below if you do, or think of something more permanent
        df = self.raw_data_df
        # df = df.loc[df["generation_number"] < 10]
        print("THIS FUNCTION SHOULD NOT EXECUTE.")
        return df

    def get_fitness_df(self):
        a = self.raw_data_df.groupby("generation_number")["individual_fitness"].mean()
        b = self.raw_data_df.groupby("generation_number")["combined_fitness"].mean()
        c = self.raw_data_df.groupby("generation_number")["collective_fitness"].mean()
        df = pd.concat([a, b, c], axis=1)
        return df

    def get_nonzero_gene_count_per_chromosome_df(self):
        df = self.raw_data_df[
            [
                "super_run_number",
                "generation_number",
                "parameter_id",
                "chromosome_id",
                "chromosome_type",
                "chromosome",
            ]
        ]
        df = df.drop_duplicates(
            ["super_run_number", "generation_number", "chromosome_id"]
        )
        df["nonzero_genes"] = df["chromosome"].apply(
            lambda chromosome: np.count_nonzero(chromosome)
        )
        df = df[["generation_number", "nonzero_genes"]]

        df = (
            df.groupby(["generation_number"])["nonzero_genes"]
            .value_counts()
            .unstack(fill_value=0)
        )
        # df = df.groupby(["generation_number"]).mean()
        return df

    def get_composition_size_df(self):
        df = self.raw_data_df[
            [
                "super_run_number",
                "generation_number",
                "parameter_id",
                "composition",
            ]
        ]
        df = df.drop_duplicates(
            ["super_run_number", "generation_number", "parameter_id"]
        )
        df["composition_size"] = df["composition"].apply(lambda x: len(x))
        df = df[["generation_number", "composition_size"]]
        df = df.groupby(["generation_number"]).mean()
        return df

    def get_species_per_composition_df(self):
        df = self.raw_data_df[
            [
                "super_run_number",
                "generation_number",
                "parameter_id",
                "parameter_run_number",
                "chromosome_id",
                "chromosome_type",
            ]
        ]
        df = (
            df.groupby(
                [
                    "super_run_number",
                    "generation_number",
                    "parameter_id",
                    "parameter_run_number",
                ]
            )["chromosome_type"]
            .value_counts()
            .unstack()
            .fillna(0)
        )
        df = df.groupby(["generation_number"]).mean()
        df = self.sort_df_colums(df)
        return df

    def get_species_per_generation_df(self):
        df = self.raw_data_df[
            [
                "super_run_number",
                "generation_number",
                "chromosome_id",
                "chromosome_type",
            ]
        ]
        df = df.drop_duplicates(
            ["super_run_number", "generation_number", "chromosome_id"]
        )
        df = (
            df.groupby(["super_run_number", "generation_number"])["chromosome_type"]
            .value_counts()
            .unstack()
            .fillna(0)
        )
        df = df.groupby(["generation_number"]).mean()
        df = self.sort_df_colums(df)
        return df

    def get_number_of_species_per_generation_df(self):
        df = self.raw_data_df[
            [
                "super_run_number",
                "generation_number",
                "chromosome_type",
            ]
        ]
        df = (
            df.groupby(["super_run_number", "generation_number"])["chromosome_type"]
            .nunique()
            .reset_index()
        )
        df = df.rename(columns={"chromosome_type": "number_of_species"})
        df = df.drop(["super_run_number"], axis=1)
        df = df.groupby(["generation_number"]).mean()
        return df

    def get_species_fitness_df(self):
        df = self.raw_data_df[
            [
                "super_run_number",
                "generation_number",
                "average_fitness_per_species",
            ]
        ]
        df = df.drop_duplicates(["super_run_number", "generation_number"])
        df = pd.concat(
            [
                df["generation_number"],
                df["average_fitness_per_species"].apply(pd.Series),
            ],
            axis=1,
        )
        df = df.groupby(["generation_number"]).mean()
        df = self.sort_df_colums(df)
        return df

    def get_species_representation_coefficient_df(self):
        composition_df = self.get_species_per_composition_df()
        population_df = self.get_species_per_generation_df()
        composition_total_vector = composition_df.sum(axis=1)
        population_total_vector = population_df.sum(axis=1)
        scale_vector = population_total_vector / composition_total_vector
        composition_df = composition_df * scale_vector.values[:, None]
        # print(f"composition_df: {composition_df}")
        # print(f"population_df: {population_df}")
        np.seterr(invalid="ignore")
        df = pd.DataFrame(
            population_df.values / composition_df.values,
            index=composition_df.index,
            columns=composition_df.columns,
        )
        np.seterr(invalid="warn")
        # print(f"representation_df: {df}")
        return df

    def get_composition_diversity_fitness_correlation_df(self):
        # NOTE TODO This function needs to be checked over extremely carefully. Especially the fitness part.
        # Note checked.
        df = self.raw_data_df[
            [
                "super_run_number",
                "generation_number",
                "parameter_id",
                "parameter_run_number",
                "collective_fitness",
                "chromosome_type",
            ]
        ]
        df = (
            df.groupby(
                [
                    "super_run_number",
                    "generation_number",
                    "parameter_id",
                    "parameter_run_number",
                ]
            )["chromosome_type"]
            .value_counts()
            .unstack()
            .fillna(0)
        )
        df["shannon_diversity"] = df.apply(self.shannon_diversity, axis=1)
        df["collective_fitness"] = self.raw_data_df.groupby(
            [
                "super_run_number",
                "generation_number",
                "parameter_id",
                "parameter_run_number",
            ]
        )["collective_fitness"].mean()

        df = df.reset_index()[["shannon_diversity", "collective_fitness"]]
        return df

    def shannon_diversity(self, row):
        total = row.sum()
        proportions = row / total
        with np.errstate(divide="ignore"):
            shannon_div = -np.sum(proportions * np.log(proportions))
        return shannon_div if not np.isnan(shannon_div) else 0

    def get_benchmark_df(self):
        df = self.raw_data_df[
            [
                "super_run_number",
                "generation_number",
                "benchmark_score",
            ]
        ]
        df = df.dropna(subset=["benchmark_score"])
        df = df.drop_duplicates(["super_run_number", "generation_number"])
        df = df.groupby(["generation_number"]).mean().reset_index()
        return df

    def get_species_fitness_correlation(self, fitness_type: str = "collective_fitness"):
        # Calculate the fitness correlation between types in a composition --
        # How well does this species perform with these other species? --
        # dimensions are the number of species, indices are the species,
        # values are the average fitness of the species with the other species

        print("Unimplemented method called.")

        df = self.raw_data_df[
            [
                "super_run_number",
                "generation_number",
                "parameter_id",
                "parameter_run_number",
                "chromosome_type",
                fitness_type,
            ]
        ]

    def get_final_chromosomes_df(self):
        df = self.raw_data_df[
            [
                "super_run_number",
                "generation_number",
                "chromosome_id",
                "chromosome_type",
                "chromosome",
                "combined_fitness",
            ]
        ]
        df = df.loc[df["super_run_number"] == 0]
        df = df.loc[df["generation_number"] == df["generation_number"].max()]
        df = df.drop_duplicates(["chromosome_id"])
        df = df[["chromosome_type", "chromosome", "combined_fitness"]]
        df = df.set_index("chromosome_type")
        return df

    def get_chromosomes_cluster_df(self, cluster_method: str):
        df = self.get_final_chromosomes_df()
        chromosome_length = len(df.iloc[0]["chromosome"])
        df[[i for i in range(0, chromosome_length)]] = pd.DataFrame(
            df.chromosome.tolist(), index=df.index
        )
        fitness_df = df["combined_fitness"]
        df = df.drop(columns=["chromosome", "combined_fitness"])
        if cluster_method == "t-sne":
            chromosome_positions = TSNE(
                n_components=2,
                random_state=0,
                perplexity=25,
                init="random",
                learning_rate="auto",
            ).fit_transform(df)
        elif cluster_method == "umap":
            chromosome_positions = UMAP(n_components=2, random_state=0).fit_transform(
                df
            )
        else:
            raise ValueError(
                f"cluster_method must be one of ['t-sne', 'umap'], not {cluster_method}"
            )
        cluster_df = pd.DataFrame(chromosome_positions, columns=["x", "y"])
        cluster_df["chromosome_type"] = df.index
        fitness_df = fitness_df.reset_index()
        cluster_df["fitness"] = fitness_df["combined_fitness"]
        return cluster_df
