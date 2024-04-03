from dataclasses import dataclass
from shared_components.parameters import Parameters
from speciation.chromosome import Chromosome


@dataclass
class BenchmarkFunctionParameters(Parameters):
    function_name: str
    n_agents: int
    agent_chromosomes: list[Chromosome]
    agent_mode: str

    collective_fitness_type: str
    collective_fitness_weight: float
    gene_cost_fitness_penalty: float
    gene_cost_multiplier: float

    def from_config(config):
        gene_value_boundaries = config[config["FUNCTION_NAME"]]["GENE_VALUE_BOUNDARIES"]
        return BenchmarkFunctionParameters(
            n_agents=config["N_AGENTS"],
            function_name=config["FUNCTION_NAME"],
            agent_chromosomes=[
                Chromosome.generate_random_with_range(
                    config["N_GENES"],
                    gene_value_boundaries[0],
                    gene_value_boundaries[1],
                )
                for _ in range(config["N_AGENTS"])
            ],
            agent_mode=config["AGENT_MODE"],
            collective_fitness_type=config["COLLECTIVE_FITNESS_TYPE"],
            collective_fitness_weight=config["COLLECTIVE_FITNESS_WEIGHT"],
            gene_cost_fitness_penalty=config["GENE_COST"],
            gene_cost_multiplier=config["GENE_COST_MULTIPLIER"],
        )
