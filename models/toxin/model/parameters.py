from dataclasses import dataclass
from shared_components.parameters import Parameters
from speciation.chromosome import Chromosome


@dataclass
class ToxinParameters(Parameters):
    max_steps: int
    n_agents: int
    n_toxins: int

    toxin_cleanup_rate: int
    gene_cost: int
    gene_cost_multiplier: float

    toxin_base: int

    collective_fitness_type: str
    collective_fitness_weight: float

    agent_chromosomes: list[Chromosome]

    def from_config(config):
        return ToxinParameters(
            max_steps=config["MAX_STEPS"],
            n_agents=config["N_AGENTS"],
            n_toxins=config["N_TOXINS"],
            toxin_cleanup_rate=config["TOXIN_CLEANUP_RATE"],
            gene_cost=config["GENE_COST"],
            gene_cost_multiplier=config["GENE_COST_MULTIPLIER"],
            toxin_base=config["TOXIN_BASE"],
            collective_fitness_type=config["COLLECTIVE_FITNESS_TYPE"],
            collective_fitness_weight=config["COLLECTIVE_FITNESS_WEIGHT"],
            agent_chromosomes=[
                Chromosome.generate_random(config["N_TOXINS"])
                for _ in range(config["N_AGENTS"])
            ],
        )
