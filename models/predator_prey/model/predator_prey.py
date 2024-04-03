from dataclasses import dataclass

from models.predator_prey.model.parameters import PPParameters
from .agent import Hunter, MLPHunter, Prey
from .event_manager import TickEvent, QuitEvent
from shared_components.model_data import ModelData
from shared_components.model_interface import Model
from shared_components.logger import log
import pandas as pd


@dataclass
class PPData(ModelData):
    ticks_elapsed: int
    agent_data: list
    composition: list

    def to_dataframe(self):
        df = pd.DataFrame(
            {
                "model_name": [self.model_name] * len(self.agent_data),
                "composition": [self.composition] * len(self.agent_data),
            }
        )
        for column in self.agent_data[0].keys():
            df[column] = [hunter[column] for hunter in self.agent_data]
        return df


class PredatorPrey(Model):
    def __init__(
        self,
        parameters: PPParameters,
        event_manager=None,
        run_with_viewer=False,
        frames_per_second=60,
    ):
        self.run_with_viewer = run_with_viewer
        self.field_size_x = parameters.field_size_x
        self.field_size_y = parameters.field_size_y

        if self.run_with_viewer and event_manager != None:
            import pygame

            pygame.init()
            self.event_manager = event_manager
            self.event_manager.register_listener(self)
            self.fps = frames_per_second
            self.clock = pygame.time.Clock()

        self.parameters = parameters

        self.ticks = 0
        self.max_ticks = parameters.max_ticks
        self.stop_on_kill = parameters.model_stop_on_kill
        self.done = False

        self.use_mlp_hunters = parameters.use_mlp_predators
        self.predator_chromosomes = parameters.agent_chromosomes

        self.n_hunters = parameters.n_predators
        self.n_prey = parameters.n_prey

        self.hunters = []
        self.prey = []

        self.agent_args = {
            "field_size_x": parameters.field_size_x,
            "field_size_y": parameters.field_size_y,
            "toroidal_space": parameters.toroidal_space,
        }

        self.hunter_fitness_function = self._get_hunter_fitness_function()

        self.model_data = PPData(
            model_name="Predator_Prey", ticks_elapsed=0, agent_data=[], composition=[]
        )
        self.debug_values = {}
        if self.use_mlp_hunters and len(self.predator_chromosomes) != self.n_hunters:
            log.warning(
                f"Number of predators ({self.n_hunters}) and chromosomes ({len(self.predator_chromosomes)}) do not match. Try checking the chromosome types, chromosome sampling, and number of predators/agents."
            )
        self.init_model()

    def init_model(self):
        self.init_hunters()
        self.init_prey()
        self.set_hunter_targets()
        self.set_prey_chasers()

    def reset_model(self):
        self.init_model()
        self.ticks = 0
        self.done = False

    def init_hunters(self):
        # This function should be refactored with the factory pattern
        self.hunters = []

        if self.use_mlp_hunters:
            for chromosome in self.predator_chromosomes:
                hunter = MLPHunter(
                    **self.agent_args,
                    predator_respawn_on_kill=self.parameters.predator_respawn_on_kill,
                    chromosome=chromosome,
                )
                self.hunters.append(hunter)
        else:
            for _ in range(self.n_hunters):
                hunter = Hunter(
                    **self.agent_args,
                    predator_respawn_on_kill=self.parameters.predator_respawn_on_kill,
                )
                self.hunters.append(hunter)

        for hunter in self.hunters:
            if self.parameters.predator_random_spawn:
                hunter.set_random_position()
            else:
                hunter.set_position((0, 0))

    def init_prey(self):
        self.prey = []
        for _ in range(self.n_prey):
            prey = Prey(
                **self.agent_args,
                prey_behaviour=self.parameters.prey_behaviour,
                prey_respawn_on_death=self.parameters.prey_respawn_on_death,
            )
            if self.parameters.prey_random_spawn:
                prey.set_random_position()
            self.prey.append(prey)

    def set_hunters(self, chromosomes):
        self.predator_chromosomes = chromosomes
        self.init_hunters()

    def set_hunter_targets(self):
        for hunter in self.hunters:
            hunter.set_target(self.prey[0])

    def set_prey_chasers(self):
        for prey in self.prey:
            prey.set_chasers(self.hunters)

    def notify(self, event):
        if isinstance(event, QuitEvent):
            self.done = True

    def tick_entities(self):
        for hunter in self.hunters:
            hunter.tick()
        for prey in self.prey:
            prey.tick()

    def tick(self):
        # model changes per tick
        self.tick_entities()

        if self.run_with_viewer:
            self.clock.tick(self.fps)
            self.event_manager.post(TickEvent())
        self.ticks += 1

    def check_stopping_criterion(self):
        if self.max_ticks != None and self.ticks >= self.max_ticks:
            self.done = True
        elif self.stop_on_kill:
            self.done = self.is_prey_caught()

    def is_prey_caught(self) -> bool:
        for hunter in self.hunters:
            if hunter.kills == 1:
                return True
        return False

    def run(self):
        while not self.done:
            self.tick()
            self.check_stopping_criterion()
        if self.run_with_viewer:
            pygame.quit()

    def run_benchmark(self) -> int:
        return self.run_benchmark_trails()

    def run_benchmark_trails(self) -> int:
        self.stop_on_kill = True
        prey_positions = [
            ((self.field_size_x / 6) * x, (self.field_size_y / 6) * y)
            for x in range(1, 7, 2)
            for y in range(1, 7, 2)
        ]
        succeses = 0
        for prey_pos in prey_positions:
            succeses += self.run_benchmark_trail(prey_pos)
        return succeses

    def run_benchmark_trail(self, prey_pos):
        self.reset_model()
        for hunter in self.hunters:
            hunter.set_position((0, 0))
        for prey in self.prey:
            prey.set_position(prey_pos)
        while not self.done:
            self.tick()
            self.check_stopping_criterion()
        return self.is_prey_caught()

    def _get_hunter_fitness_function(self):
        if self.parameters.individual_fitness_type == "individual_kills":
            return self._hunter_fitness_kills
        elif self.parameters.individual_fitness_type == "distance_to_prey":
            return self._hunter_fitness_distance_to_prey
        elif self.parameters.individual_fitness_type == "distance_to_prey_greedy":
            return self._hunter_fitness_distance_to_prey_greedy
        elif self.parameters.individual_fitness_type == "distance_to_prey_positive":
            return self._hunter_fitness_distance_to_prey_positive
        else:
            raise NotImplementedError(
                f"Unknown individual fitness type {self.parameters.individual_fitness_type}"
            )

    def _hunter_fitness_kills(self, hunter):
        return hunter.kills

    def _hunter_fitness_distance_to_prey(self, hunter):
        if self.is_prey_caught():
            return (1000 - hunter.distance_to_target) / 10
        else:
            return (hunter.initial_distance_to_target - hunter.distance_to_target) / 10

    def _hunter_fitness_distance_to_prey_greedy(self, hunter):
        if hunter.kills >= 1:
            return (1000 - hunter.distance_to_target) / 10
        else:
            return (hunter.initial_distance_to_target - hunter.distance_to_target) / 10

    def _hunter_fitness_distance_to_prey_positive(self, hunter):
        if self.is_prey_caught():
            return (354 + 10000 - hunter.distance_to_target) / 10
        else:
            return (
                354 + hunter.initial_distance_to_target - hunter.distance_to_target
            ) / 10

    def calculate_individual_fitness(self, hunter):
        return self.hunter_fitness_function(hunter)

    def calculate_collective_fitness(self):
        collective_fitness = sum(
            [self.hunter_fitness_function(hunter) for hunter in self.hunters]
        ) / len(self.hunters)
        return collective_fitness

    def calculate_combined_fitness(self, individual_fitness, collective_fitness):
        combined_fitness = (
            collective_fitness * self.parameters.collective_fitness_weight
        ) + (individual_fitness * (1.0 - self.parameters.collective_fitness_weight))
        return combined_fitness

    def get_data(self):
        self.model_data.ticks_elapsed = self.ticks
        # This function might be doing too many things. I do want to calculate the collective fitness on the model level.
        # alternative is to have hunters have a collective_fitness value that is filled in on another function, but that seems
        # like a purely architectural change. For now, this is fine. (words spoken on many permanent temporary solutions)
        collective_fitness = self.calculate_collective_fitness()
        for hunter in self.hunters:
            hunter_data = hunter.get_data()
            individual_fitness = self.calculate_individual_fitness(hunter)
            combined_fitness = self.calculate_combined_fitness(
                individual_fitness=individual_fitness,
                collective_fitness=collective_fitness,
            )
            hunter_data["individual_fitness"] = individual_fitness
            hunter_data["collective_fitness"] = collective_fitness
            hunter_data["combined_fitness"] = combined_fitness
            self.model_data.agent_data.append(hunter_data)
        self.model_data.composition = [
            hunter.brain.chromosome.chromosome_type for hunter in self.hunters
        ]
        return self.model_data
