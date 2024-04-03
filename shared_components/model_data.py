import pandas as pd
from dataclasses import dataclass
from .logger import log


@dataclass
class ModelData:
    # Interface class for model data. Each Model implements its own ModelData class.
    # Captures the data about that specific model run.
    model_name: str

    def to_dataframe(self):
        log.warning("Creating dataframe from ModelData interface object.")
        df = pd.DataFrame({"model_name": self.model_name})
        return df


@dataclass
class ModelRunData:
    # Data from a single model run, including metadata about which parameter and # of run of that parameter
    # A model run for a parameter is a model run for a single permutation of chromosomes.
    total_run_number: int
    parameter_run_number: int
    parameter_id: int
    model_data: ModelData

    def to_dataframe(self):
        # There might be a simpler way to do this. I'm eager to learn it.
        df = self.model_data.to_dataframe()
        df["total_run_number"] = [self.total_run_number] * len(df.index)
        df["parameter_run_number"] = [self.parameter_run_number] * len(df.index)
        df["parameter_id"] = [self.parameter_id] * len(df.index)
        return df


@dataclass
class ModelGenerationData:
    # Data from a single generation of a model run, including generation metadata, and a list of all model runs in that generation
    # A generation has model runs all drawn from the same pool of chromsomes, but in different permutations.
    generation_number: int
    best_fitness_per_species: float
    average_fitness_per_species: float
    benchmark_score: float
    benchmark_best_combination: str
    benchmark_best_combination_score: float
    benchmark_best_combination_count: int
    benchmark_worst_combination: str
    benchmark_worst_combination_score: float
    generation_model_run_data: list[ModelRunData]

    def to_dataframe(self):
        df = pd.concat(
            [
                model_run_data.to_dataframe()
                for model_run_data in self.generation_model_run_data
            ]
        )
        df["generation_number"] = [self.generation_number] * len(df.index)
        df["best_fitness_per_species"] = [self.best_fitness_per_species] * len(df.index)
        df["average_fitness_per_species"] = [self.average_fitness_per_species] * len(
            df.index
        )
        df["benchmark_score"] = [self.benchmark_score] * len(df.index)
        df["benchmark_best_combination"] = [self.benchmark_best_combination] * len(
            df.index
        )
        df["benchmark_best_combination_score"] = [
            self.benchmark_best_combination_score
        ] * len(df.index)
        df["benchmark_best_combination_count"] = [
            self.benchmark_best_combination_count
        ] * len(df.index)
        df["benchmark_worst_combination"] = [self.benchmark_worst_combination] * len(
            df.index
        )
        df["benchmark_worst_combination_score"] = [
            self.benchmark_worst_combination_score
        ] * len(df.index)
        return df


@dataclass
class ModelSuperRunData:
    # Data from a single super run of a model, including super run metadata, and a list of all generations in that super run
    # A super run constitutes a single experiment starting from tabula rasa until stopping criterion.
    super_run_number: int
    generation_data: list[ModelGenerationData]

    def to_dataframe(self):
        df = pd.concat(
            [generation_data.to_dataframe() for generation_data in self.generation_data]
        )
        df["super_run_number"] = [self.super_run_number] * len(df.index)
        return df
