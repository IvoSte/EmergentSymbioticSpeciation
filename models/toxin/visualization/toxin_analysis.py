from math import ceil
import numpy as np
import pandas as pd
from shared_components.model_data import ModelSuperRunData
from shared_components.model_analysis import ModelAnalysis

# raw_data_df = ['model_name', 'toxin_concentration', 'id', 'chromosome',
#        'chromosome_id', 'chromosome_type', 'active_genes', 'inactive_genes',
#        'individual_fitness', 'gene_fitness_cost', 'toxin_fitness_cost',
#        'collective_fitness', 'combined_fitness', 'total_run_number',
#        'parameter_run_number', 'parameter_id', 'generation_number',
#        'best_fitness_per_species', 'average_fitness_per_species',
#        'benchmark_score', 'benchmark_best_combination',
#        'benchmark_best_combination_score', 'benchmark_best_combination_count',
#        'benchmark_worst_combination', 'benchmark_worst_combination_score',
#        'super_run_number'],


class ToxinAnalysis(ModelAnalysis):
    def __init__(self, data: list[ModelSuperRunData], parameters):
        super().__init__(data, parameters)

    def expected_equilibrium_cooperators(self):
        expected_equilibrium_cooperators = (
            self.parameters.toxin_base - self.parameters.gene_cost
        ) / self.parameters.toxin_cleanup_rate
        return expected_equilibrium_cooperators

    def expected_equilibrium_fitness(self):
        equilibrium_cooperators = self.expected_equilibrium_cooperators()
        cooperator_fitness = self.parameters.gene_cost
        cooperators_fitness = equilibrium_cooperators * cooperator_fitness

        defector_fitness = max(
            0,
            self.parameters.toxin_base
            - (equilibrium_cooperators * self.parameters.toxin_cleanup_rate),
        )
        defectors_fitness = (
            self.parameters.n_agents - equilibrium_cooperators
        ) * defector_fitness

        expected_equilibrium_fitness = (
            -((cooperators_fitness + defectors_fitness) / self.parameters.n_agents)
            * self.parameters.n_toxins
        )

        return expected_equilibrium_fitness

    def get_average_stat_per_generation(self, stat_name):
        assert (
            stat_name in self.raw_data_df.columns
        ), f"Stat {stat_name} not in raw data df"
        stat_df = self.raw_data_df.groupby("generation_number").mean()[stat_name]
        return pd.DataFrame(stat_df)

    def get_fitness_df(self):
        a = self.raw_data_df.groupby("generation_number")["individual_fitness"].mean()
        b = self.raw_data_df.groupby("generation_number")["toxin_fitness_cost"].mean()
        c = self.raw_data_df.groupby("generation_number")["gene_fitness_cost"].mean()
        df = pd.merge(a, b, on="generation_number", how="outer")
        df = pd.merge(df, c, on="generation_number", how="outer")
        return df

    def get_genes_df(self):
        # From the list of active genes for each agent, create a dataframe with the number of agents with each gene
        # per generation, averaged across runs
        df = self.raw_data_df[
            ["super_run_number", "generation_number", "parameter_id", "active_genes"]
        ]
        # explode the list of active genes into a row for each gene
        df = df.explode("active_genes")
        # adding dummy gene in place of NaNs to preserve generations with no active genes
        df = df.fillna("dummy")

        df = (
            df.groupby(
                [
                    "super_run_number",
                    "generation_number",
                    "parameter_id",
                    "active_genes",
                ]
            )
            .size()
            .unstack()
        )
        if "dummy" in df.columns:
            df = df.drop(["dummy"], axis=1)
        df = df.groupby(["generation_number"]).mean().fillna(0.0)
        return df

    def get_agents_activity_df(self):
        # How many genes are active per agent
        df = self.raw_data_df[
            ["super_run_number", "generation_number", "parameter_id", "active_genes"]
        ].copy()
        df["n_active_genes"] = df["active_genes"].apply(lambda x: len(x))

        df = (
            df.groupby(["super_run_number", "generation_number", "parameter_id"])[
                "n_active_genes"
            ]
            .value_counts()
            .reset_index(name="count")
            .pivot(
                columns=["n_active_genes"],
                index=["super_run_number", "generation_number", "parameter_id"],
                values="count",
            )
        )
        df = df.fillna(0)
        df = df.groupby(["generation_number"]).mean()
        return df

    def get_toxin_df(self):
        # The toxin concentration per generation
        df = self.raw_data_df[
            [
                "super_run_number",
                "generation_number",
                "parameter_id",
                "toxin_concentration",
            ]
        ]
        df = df.drop_duplicates(
            ["super_run_number", "generation_number", "parameter_id"]
        )
        df[[i for i in range(self.parameters.n_toxins)]] = pd.DataFrame(
            df.toxin_concentration.tolist(), index=df.index
        )
        df = df.drop(
            columns=["toxin_concentration", "super_run_number", "parameter_id"]
        )
        df = df.fillna(0)
        df = df.groupby(["generation_number"]).mean()
        return df

    def get_redundant_genes_df(self):
        # How many agents have redundant genes -- the number of genes active above the required number of genes to clean up all toxin
        df = self.raw_data_df[
            ["super_run_number", "generation_number", "parameter_id", "active_genes"]
        ]
        df = df.explode("active_genes")
        df = df.fillna("dummy")

        df = (
            df.groupby(
                [
                    "super_run_number",
                    "generation_number",
                    "parameter_id",
                    "active_genes",
                ]
            )
            .size()
            .unstack()
        )
        n_genes_required = ceil(
            int(self.parameters.toxin_base / self.parameters.toxin_cleanup_rate)
        )
        df = df.map(lambda x: x - n_genes_required if x >= n_genes_required else 0)
        if "dummy" in df.columns:
            df = df.drop(["dummy"], axis=1)
        df = df.groupby(["generation_number"]).mean()
        return df
