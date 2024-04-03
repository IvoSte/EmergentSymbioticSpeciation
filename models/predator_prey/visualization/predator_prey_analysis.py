import pandas as pd
from shared_components.model_data import ModelSuperRunData
from shared_components.model_analysis import ModelAnalysis

# 'model_name', 'name', 'kills', 'individual_fitness', 'chromosome',
#        'chromosome_type', 'chromosome_id', 'collective_fitness',
#        'combined_fitness', 'total_run_number', 'parameter_run_number',
#        'parameter_id', 'generation_number', 'best_fitness_per_species',
#        'average_fitness_per_species', 'benchmark_score',
#        'benchmark_best_combination', 'benchmark_best_combination_score',
#        'benchmark_best_combination_count', 'benchmark_worst_combination',
#        'benchmark_worst_combination_score', 'super_run_number']


class PredatorPreyAnalysis(ModelAnalysis):
    def __init__(self, data: list[ModelSuperRunData], parameters):
        super().__init__(data, parameters)

    def get_fitness_df(self):
        a = self.raw_data_df.groupby("generation_number")["individual_fitness"].mean()
        b = self.raw_data_df.groupby("generation_number")["collective_fitness"].mean()
        c = self.raw_data_df.groupby("generation_number")["combined_fitness"].mean()
        df = pd.concat([a, b, c], axis=1)
        return df

    def get_catches_per_species_df(self):
        df = self.raw_data_df[
            [
                "super_run_number",
                "generation_number",
                "parameter_id",
                "parameter_run_number",
                "chromosome_type",
                "kills",
            ]
        ]
        df = (
            df.groupby(
                [
                    "super_run_number",
                    "generation_number",
                    "parameter_id",
                    "parameter_run_number",
                    "chromosome_type",
                ]
            )["kills"]
            .sum()
            .unstack()
        )
        df = df.groupby(["generation_number"]).mean()
        df["total"] = df.sum(axis=1)
        df = self.sort_df_colums(df)

        return df

    # def save_best_chromosomes(self):
    #     df = self.raw_data_df[
    #         [
    #             "super_run_number",
    #             "generation_number",
    #             "parameter_id",
    #             "parameter_run_number",
    #             "chromosome_type",
    #             "chromosome_id",
    #             "chromosome",
    #             "individual_fitness",
    #             "collective_fitness",
    #             "combined_fitness",
    #         ]
    #     ]
    #     df = df.groupby(
    #         [
    #             "super_run_number",
    #             "generation_number",
    #             "parameter_id",
    #             "parameter_run_number",
    #             "chromosome_type",
    #         ]
    #     ).apply(lambda x: x.nlargest(1, "combined_fitness"))
    #     df = df.reset_index(drop=True)
    #     df.to_csv("best_chromosomes.csv", index=False)
    #     return df
