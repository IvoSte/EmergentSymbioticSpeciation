from dataclasses import dataclass
from models.function_optimization.model.optimization_functions import (
    BenchmarkFunctionFactory,
)
from shared_components.model_data import ModelData
from shared_components.model_interface import Model
from .agent import Agent
import numpy as np
import pandas as pd


@dataclass
class BenchmarkFunctionModelData(ModelData):
    function_name: str
    agent_data: list
    solution_vector: np.ndarray
    objective_value: float
    composition: list

    def to_dataframe(self):
        df = pd.DataFrame(
            {
                "model_name": [self.model_name] * len(self.agent_data),
                "function_name": [self.function_name] * len(self.agent_data),
                "solution_vector": [self.solution_vector] * len(self.agent_data),
                "objective_value": [self.objective_value] * len(self.agent_data),
                "composition": [self.composition] * len(self.agent_data),
            }
        )
        for column in self.agent_data[0].keys():
            df[column] = [agent[column] for agent in self.agent_data]
        return df


class BenchmarkFunctionModel(Model):
    def __init__(self, parameters):
        self.parameters = parameters
        self.function_name = parameters.function_name
        self.benchmark_function = BenchmarkFunctionFactory.create_function(
            self.function_name
        )
        self.agent_mode = parameters.agent_mode

        self.agents = self._init_agents(self.parameters.agent_chromosomes)
        self.solution_vector = self._calculate_solution_vector()
        self.function_evaluation = None

        if self.agent_mode == "partial size":
            self.n_dimensions = parameters.n_agents * len(
                parameters.agent_chromosomes[0].genes
            )
            assert (
                self.solution_vector.shape[0] == self.n_dimensions
            ), f"Number of dimensions ({self.n_dimensions}) must match number of agents ({self.parameters.n_agents}) * genes per agent ({len(parameters.agent_chromosomes[0].genes)}) or be None"

        elif self.agent_mode == "full size":
            self.n_dimensions = len(parameters.agent_chromosomes[0].genes)
            assert (
                self.solution_vector.shape[0] == self.n_dimensions
            ), f"Number of dimensions ({self.n_dimensions}) must match number of genes per agent ({len(parameters.agent_chromosomes[0].genes)}) or be None"

    def _init_agents(self, agent_chromosomes):
        return [
            Agent(
                chromosome=chromosome,
                gene_cost=self.parameters.gene_cost_fitness_penalty,
                gene_cost_multiplier=self.parameters.gene_cost_multiplier,
            )
            for chromosome in agent_chromosomes
        ]

    def run(self):
        self._step()
        self._inform_agents_of_result()
        # self._print_report()

    def _calculate_solution_vector(self):
        if self.agent_mode == "partial size":
            return np.array([value for agent in self.agents for value in agent.values])
        elif self.agent_mode == "full size":
            return np.sum([agent.values for agent in self.agents], axis=0)

    def _step(self):
        self.function_evaluation = self._solve(
            self.benchmark_function, self.solution_vector
        )

    def _solve(self, benchmark_function, solution_vector):
        return benchmark_function.evaluate(solution_vector)

    def _calculate_collective_fitness(self):
        assert (
            self.function_evaluation is not None
        ), "Must run model before calculating collective fitness"
        if self.parameters.collective_fitness_type == "objective value":
            return -self.function_evaluation
        elif self.parameters.collective_fitness_type == "average":
            assert (
                self.agents[0].fitness is not None
            ), "Must calculate individual fitness before calculating collective fitness, if using average fitness as collective fitness."
            return np.mean([agent.fitness for agent in self.agents])

    def _calculate_individual_fitness(self, agent):
        penalty = (
            len([value for value in agent.values if value != 0.0])
            * self.parameters.gene_cost_fitness_penalty
        )
        return -self.function_evaluation - penalty

    def _inform_agents_of_result(self):
        for agent in self.agents:
            agent.set_objective_value(self.function_evaluation)

    def calculate_combined_fitness(self, individual_fitness, collective_fitness):
        combined_fitness = (
            collective_fitness * self.parameters.collective_fitness_weight
        ) + (individual_fitness * (1.0 - self.parameters.collective_fitness_weight))
        return combined_fitness

    def get_data(self):
        agent_data = [agent.get_data() for agent in self.agents]
        collective_fitness = self._calculate_collective_fitness()
        # print(f"Solution: {self.solution}")
        # print(f"Fitness: {fitness}")
        for agent in agent_data:
            agent["collective_fitness"] = collective_fitness
            agent["combined_fitness"] = self.calculate_combined_fitness(
                agent["individual_fitness"], collective_fitness
            )
        composition = [agent.chromosome.chromosome_type for agent in self.agents]

        model_data = BenchmarkFunctionModelData(
            model_name="BenchmarkFunctionModel",
            function_name=self.function_name,
            agent_data=agent_data,
            solution_vector=self.solution_vector,
            objective_value=self.function_evaluation,
            composition=composition,
        )

        return model_data

    def _print_report(self):
        for idx, agent in enumerate(self.agents):
            print(f"Agent {idx} {agent.chromosome.chromosome_type}: {agent.values}")
        print(f"Solution: {self.solution_vector}")
        print(f"Objective value: {self.function_evaluation}")
        print("\n")
