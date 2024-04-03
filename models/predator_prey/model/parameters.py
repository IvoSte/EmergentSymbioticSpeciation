from dataclasses import dataclass
from shared_components.parameters import Parameters
from speciation.chromosome import Chromosome


@dataclass
class PPParameters(Parameters):
    max_ticks: int
    n_predators: int
    n_prey: int
    use_mlp_predators: bool
    agent_chromosomes: list[Chromosome]
    predator_random_spawn: bool
    prey_random_spawn: bool
    predator_respawn_on_kill: bool
    prey_respawn_on_death: bool
    model_stop_on_kill: bool
    prey_behaviour: str
    collective_fitness_type: str
    individual_fitness_type: str
    collective_fitness_weight: float

    field_size_x: int
    field_size_y: int
    toroidal_space: bool

    def from_config(config):
        return PPParameters(
            max_ticks=config["MAX_TICKS"],
            n_predators=config["N_PREDATORS"],
            n_prey=config["N_PREY"],
            predator_random_spawn=config["PREDATOR_RANDOM_SPAWN"],
            prey_random_spawn=config["PREY_RANDOM_SPAWN"],
            predator_respawn_on_kill=config["PREDATOR_RESPAWN_ON_KILL"],
            prey_respawn_on_death=config["PREY_RESPAWN_ON_DEATH"],
            model_stop_on_kill=config["MODEL_STOP_ON_KILL"],
            collective_fitness_type=config["COLLECTIVE_FITNESS_TYPE"],
            individual_fitness_type=config["INDIVIDUAL_FITNESS_TYPE"],
            collective_fitness_weight=config["COLLECTIVE_FITNESS_WEIGHT"],
            use_mlp_predators=config["MLP_PREDATORS"],
            agent_chromosomes=[
                Chromosome.generate_random(config["CHROMOSOME_LENGTH"])
                for _ in range(config["N_PREDATORS"])
            ],
            prey_behaviour=config["PREY_BEHAVIOUR"],
            field_size_x=config["SCREEN_SIZE_X"],
            field_size_y=config["SCREEN_SIZE_Y"],
            toroidal_space=config["TOROIDAL_SPACE"],
        )
