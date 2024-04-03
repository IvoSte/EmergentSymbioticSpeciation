from math import ceil
import numpy as np
import pandas as pd
from shared_components.model_data import ModelSuperRunData
from shared_components.model_analysis import ModelAnalysis


# The columns, so you don't have to keep looking them up.
# ['model_name', 'function_name', 'solution_vector', 'objective_value',
#        'id', 'chromosome', 'chromosome_id', 'chromosome_type', 'agent_values',
#        'individual_fitness', 'collective_fitness', 'combined_fitness',
#        'total_run_number', 'parameter_run_number', 'parameter_id',
#        'generation_number', 'best_fitness_per_species',
#        'average_fitness_per_species', 'benchmark_score',
#        'benchmark_best_combination', 'benchmark_best_combination_score',
#        'benchmark_best_combination_count', 'benchmark_worst_combination',
#        'benchmark_worst_combination_score', 'super_run_number']


class FunctionOptimizationAnalysis(ModelAnalysis):
    def __init__(self, data: list[ModelSuperRunData], parameters):
        super().__init__(data, parameters)

    def average_objective_value_per_generation(self):
        df = self.raw_data_df.groupby(["generation_number"])["objective_value"].mean()
        return pd.DataFrame(df)

    def best_objective_value_per_generation(self):
        df = self.raw_data_df.groupby("generation_number")["objective_value"].max()
        return pd.DataFrame(df)

    def worst_objective_value_per_generation(self):
        df = self.raw_data_df.groupby("generation_number")["objective_value"].min()
        return pd.DataFrame(df)

    def get_solution_vector_and_objective_values(self):
        df = self.raw_data_df[["solution_vector", "objective_value"]]
        df["solution_vector"] = df["solution_vector"].apply(tuple)
        df = df.drop_duplicates(subset=["solution_vector"])
        # split the solution vector into two columns
        # code smell note: this is to plot a solution vector with two dimensions
        df["solution_vector_x"] = df["solution_vector"].apply(lambda x: x[0])
        df["solution_vector_y"] = df["solution_vector"].apply(lambda x: x[1])
        df = df.drop(columns=["solution_vector"])
        return df

    def test(self):
        df = self.raw_data_df[
            [
                "generation_number",
                "agent_value",
                "objective_value",
                "chromosome_type",
                "chromosome_id",
                "total_run_number",
                "parameter_run_number",
                "parameter_id",
                "super_run_number",
                "individual_fitness",
                "solution_vector",
            ]
        ]
        print(df)
