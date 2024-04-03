from dataclasses import dataclass
from .environment import Environment
from .agent import Agent
import numpy as np
from collections import defaultdict
import pandas as pd
from shared_components.model_data import ModelData
from shared_components.model_interface import Model

# @dataclass
# class ModelData:
#     model_name: str

#     def to_dataframe(self):
#         df = pd.DataFrame({"model_name": self.model_name})
#         return df


@dataclass
class ToxinData(ModelData):
    avg_fitness: dict
    active_genes: dict
    agent_activity: dict
    toxin_concentration: list
    agent_data: list
    composition: list

    def to_dataframe(self):
        df = pd.DataFrame(
            {
                "model_name": [self.model_name] * len(self.agent_data),
                "toxin_concentration": [self.toxin_concentration]
                * len(self.agent_data),
                "composition": [self.composition] * len(self.agent_data),
            }
        )
        for column in self.agent_data[0].keys():
            df[column] = [agent[column] for agent in self.agent_data]
        return df


class ToxinModel(Model):
    def __init__(self, parameters):
        assert (
            parameters.agent_chromosomes is None
            or len(parameters.agent_chromosomes) == parameters.n_agents
        ), "Number of chromosomes must match number of agents or be None"
        self.parameters = parameters
        self.n_agents = parameters.n_agents
        self.n_toxins = parameters.n_toxins
        self.environment = self.init_environment()
        self.agents = self.init_agents(self.parameters.agent_chromosomes)
        self.max_steps = parameters.max_steps

    @property
    def active_genes(self):
        active_genes_list = [agent.active_genes for agent in self.agents]
        active_genes = {t_idx: 0 for t_idx in range(self.n_toxins)}
        for genes in active_genes_list:
            for gene in genes:
                active_genes[gene] += 1
        return active_genes

    @property
    def agent_activity(self):
        agent_activity = defaultdict(int)
        for agent in self.agents:
            agent_activity[len(agent.active_genes)] += 1
        return agent_activity

    @property
    def avg_fitness(self):
        return np.mean([agent.fitness for agent in self.agents])

    @property
    def total_fitness(self):
        return sum([agent.fitness for agent in self.agents])

    @property
    def toxin_remainder_fitness(self):
        return -sum(self.environment.toxins)

    @property
    def collective_fitness(self):
        if self.parameters.collective_fitness_type == "average":
            return self.avg_fitness
        elif self.parameters.collective_fitness_type == "total":
            return self.total_fitness
        elif self.parameters.collective_fitness_type == "toxin_remainder":
            return self.toxin_remainder_fitness
        elif self.parameters.collective_fitness_type == "remainder_and_average":
            return self.avg_fitness + self.toxin_remainder_fitness
        else:
            raise ValueError(
                f"collective_fitness_type must be 'average' or 'sum', was {self.parameters.collective_fitness_type}"
            )

    @property
    def avg_gene_cost(self):
        return np.mean([agent.gene_fitness_cost for agent in self.agents])

    @property
    def avg_toxin_cost(self):
        return np.mean([agent.toxin_fitness_cost for agent in self.agents])

    @property
    def benchmark_score(self):
        return -np.mean(
            [
                toxin_amount / self.parameters.toxin_base
                for toxin_amount in self.environment.toxins
            ]
        )

    def init_environment(self):
        return Environment(self.n_toxins, self.parameters.toxin_base)

    def init_agents(self, chromosomes):
        if chromosomes is not None:
            return self.create_agents(chromosomes)
        else:
            return self.create_random_agents()

    def create_random_agents(self):
        return [
            Agent(
                chromosome=self.generate_chromosome(self.n_toxins),
                environment=self.environment,
                toxin_cleanup_rate=self.parameters.toxin_cleanup_rate,
                gene_cost=self.parameters.gene_cost,
                gene_cost_multiplier=self.parameters.gene_cost_multiplier,
            )
            for _ in range(self.n_agents)
        ]

    def create_agents(self, chromosomes):
        return [
            Agent(
                chromosome=chromosome,
                environment=self.environment,
                toxin_cleanup_rate=self.parameters.toxin_cleanup_rate,
                gene_cost=self.parameters.gene_cost,
                gene_cost_multiplier=self.parameters.gene_cost_multiplier,
            )
            for chromosome in chromosomes
        ]

    def set_agents(self, chromosomes):
        self.agents = self.create_agents(chromosomes)

    def get_population_with_fitness(self):
        agent_chromosomes = [agent.chromosome for agent in self.agents]
        agent_fitnesses = self.agent_fitnesses()
        return agent_chromosomes, np.array(agent_fitnesses)

    def generate_chromosome(self, n_genes):
        return np.random.random(n_genes)

    def step(self):
        self.environment.step()
        for agent in self.agents:
            agent.step()

    def agent_fitnesses(self):
        collective_fitness = self.collective_fitness
        return [
            self.calculate_combined_fitness(agent.fitness, collective_fitness)
            for agent in self.agents
        ]

    def run(self):
        for step in range(self.max_steps):
            self.step()

    def run_benchmark(self) -> int:
        self.step()
        return self.benchmark_score

    def calculate_combined_fitness(self, individual_fitness, collective_fitness):
        combined_fitness = (
            collective_fitness * self.parameters.collective_fitness_weight
        ) + (individual_fitness * (1.0 - self.parameters.collective_fitness_weight))
        return combined_fitness

    def get_data(self):
        collective_fitness = self.collective_fitness
        agent_data = [agent.get_data() for agent in self.agents]
        for agent in agent_data:
            agent["collective_fitness"] = collective_fitness
            agent["combined_fitness"] = self.calculate_combined_fitness(
                agent["individual_fitness"], collective_fitness
            )
        composition = [agent.chromosome.chromosome_type for agent in self.agents]

        return ToxinData(
            model_name="ToxinModel",
            avg_fitness={
                "fitness": self.avg_fitness,
                "gene_cost": self.avg_gene_cost,
                "toxin_cost": self.avg_toxin_cost,
            },
            active_genes=self.active_genes,
            agent_activity=self.agent_activity,
            toxin_concentration=self.environment.toxins,
            agent_data=agent_data,
            composition=composition,
        )

    def generation_report(self, step):
        print(f"Avg fitness: {self.avg_fitness}")
        active_genes_list = [agent.active_genes for agent in self.agents]
        active_genes = {t_idx: 0 for t_idx in range(self.n_toxins)}
        for genes in active_genes_list:
            for gene in genes:
                active_genes[gene] += 1
        print(f"Active genes: {active_genes}")

        if step % 1 == 0:
            for agent in self.agents:
                print(agent.chromosome)

    def model_run_report(self):
        print(f"active genes: {self.active_genes}")
        print(f"avg fitness: {self.avg_fitness[::100]}")
