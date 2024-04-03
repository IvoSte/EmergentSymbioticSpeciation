import random
import uuid
from shared_components.logger import log
from .util.space_calculations import SpaceCalculator
from speciation.mlp import MLP


class Agent:
    def __init__(self, field_size_x, field_size_y, toroidal_space):
        self.name = uuid.uuid4()
        self.pos = (250, 250)
        self.heading = 0
        self.speed = 5
        self.space = SpaceCalculator(field_size_x, field_size_y, toroidal_space)

    def set_random_position(self):
        self.pos = self.space.get_random_position()

    def set_position(self, pos: tuple[int, int]):
        self.pos = pos

    def move(self):
        self.determine_heading()
        self.move_step()

    def determine_heading(self):
        self.move_random()

    def move_step(self):
        self.move_in_direction(self.heading, self.speed)

    def move_xy_delta(self, dx, dy):
        self.pos = self.space.get_pos_with_delta(self.pos, dx, dy)

    def move_in_direction(self, heading, distance):
        self.pos = self.space.get_pos_with_heading_distance(self.pos, heading, distance)

    def adjust_heading(self, delta):
        self.heading = (self.heading + delta) % 360

    def set_heading(self, heading):
        self.heading = heading

    def get_heading_to(self, pos):
        return self.space.get_heading_to_pos(self.pos, pos)
        # angle = atan2(self.pos[1] - pos[1], self.pos[0] - pos[0])
        # heading = degrees(angle) + 180
        # return heading

    def get_distance_to(self, pos):
        return self.space.get_distance_to_pos(self.pos, pos)
        # dx = self.pos[0] - pos[0]
        # dy = self.pos[1] - pos[1]
        # distance = sqrt((dx * dx) + (dy * dy))
        # return distance

    def move_random(self):
        if random.random() < 0.2:
            self.heading = (self.heading + random.randint(-60, 60)) % 360

    def tick(self):
        # perform action
        self.move()

    def get_data(self):
        data = {"name": self.name}
        return data


class Hunter(Agent):
    def __init__(self, predator_respawn_on_kill, **kwargs):
        Agent.__init__(self, **kwargs)
        self.name_other = "Fox"
        self.target = None
        self.kills = 0
        self.distance_to_target = 0
        self.heading_to_target = 0
        self.respawn_on_kill = predator_respawn_on_kill

    def set_target(self, target: Agent = None):
        self.target = target
        self.initial_distance_to_target = self.get_distance_to(target.pos)

    def observe_target(self):
        self.distance_to_target = self.get_distance_to(self.target.pos)
        self.heading_to_target = self.get_heading_to(self.target.pos)

    def hunt(self):
        if self.target != None:
            self.observe_target()
            self.heading = self.heading_to_target
            self.kill_target()

    def kill_target(self):
        if self.distance_to_target < 10:
            log.debug(f"Caught the target!")
            self.target.caught()
            self.kills += 1
            if self.respawn_on_kill:
                self.set_random_position()

    def tick(self):
        # perform hunter specific action
        self.hunt()
        self.move_step()

    def get_data(self):
        data = super().get_data()
        data["kills"] = self.kills
        data["individual_fitness"] = self.kills
        # data["chromosome"] = [0.0] * 22
        # data["chromosome_type"] = "Rule-based agent"
        # data["chromosome_id"] = uuid.uuid4()
        return data


class MLPHunter(Hunter):
    def __init__(self, chromosome, move_at_variable_speed=False, **kwargs):
        Hunter.__init__(self, **kwargs)
        self.brain = MLP(
            n_input=2,
            n_hidden=4,
            n_output=2,
            hidden_layers=1,
            chromosome=chromosome,
        )
        self.target_distance = self.speed
        self.move_at_variable_speed = move_at_variable_speed
        self.predator_type = chromosome.chromosome_type

    def observe_target(self):
        self.distance_to_target = self.get_distance_to(self.target.pos)

    def get_target_position_delta(self):
        return self.space.get_position_delta(self.pos, self.target.pos)

    def move_step(self):
        # Move a shorter amount than speed if the the agent desires.
        if self.move_at_variable_speed:
            distance = (
                self.target_distance
                if self.target_distance < self.speed
                else self.speed
            )
        else:
            distance = self.speed

        self.move_in_direction(self.heading, distance)

    def hunt(self):
        if self.target != None:
            self.observe_target()
            (target_x_delta, target_y_delta) = self.brain.forward_pass(
                self.get_target_position_delta()
            )
            self.heading = self.get_heading_to(
                (self.pos[0] + target_x_delta, self.pos[1] + target_y_delta)
            )
            self.target_distance = (
                self.space.get_distance(target_x_delta, target_y_delta) * 10.0
            )
            self.kill_target()

    def get_data(self):
        data = super().get_data()
        # NOTE Should the agent know about this? Seems better to abstract a bit more
        data["individual_fitness"] = self.kills
        data["chromosome"] = self.brain.chromosome.genes
        data["chromosome_type"] = self.brain.chromosome.chromosome_type
        data["chromosome_id"] = self.brain.chromosome.chromosome_id
        # NOTE this could also be calculated somewhere else. For this model this is fine I guess
        # Having fitness functions somewhere decoupled seems like a good idea when we want to tinker
        # with them
        return data


class Prey(Agent):
    def __init__(self, prey_behaviour, prey_respawn_on_death, **kwargs):
        Agent.__init__(self, **kwargs)
        self.name_other = "Rabbit"
        self.behaviour = prey_behaviour
        self.respawn_on_death = prey_respawn_on_death
        self.chasers = []
        # self.speed = 10

    def flee_from_all(self):
        # get predators direction, move opposite their average direction, weighted by distance
        average_chaser_pos_delta = self.space.get_average_position_delta(
            self.pos, [chaser.pos for chaser in self.chasers]
        )
        average_chaser_pos = (
            average_chaser_pos_delta[0] + self.pos[0],
            average_chaser_pos_delta[1] + self.pos[1],
        )
        self.heading = (self.get_heading_to(average_chaser_pos) + 180) % 360
        self.move_step()

    def flee_from_closest_chaser(self):
        closest_chaser = self.get_closest_chaser()
        self.heading = (self.get_heading_to(closest_chaser.pos) + 180) % 360
        self.move_step()

    def flee_from_closest_chaser_with_noise(self):
        closest_chaser = self.get_closest_chaser()
        self.heading = (
            self.get_heading_to(closest_chaser.pos) + 180 + random.randint(-10, 10)
        ) % 360
        self.move_step()

    def caught(self):
        if self.respawn_on_death:
            self.set_random_position()

    def get_closest_chaser(self):
        closest_chaser = min(
            self.chasers,
            key=lambda chaser: self.space.get_distance_to_pos(self.pos, chaser.pos),
        )
        return closest_chaser

    def set_chasers(self, chasers: list[Agent]):
        self.chasers = chasers

    def move_uniform_right(self):
        self.heading = 0
        self.move_step()

    def move_erratic(self):
        self.move_random()
        self.move_step()

    def tick(self):
        # perform prey specific action
        # NOTE Can be refactored to switch case, but needs python 3.10
        if self.behaviour == "stand_still":
            return
        elif self.behaviour == "move_uniform_right":
            self.move_uniform_right()
        elif self.behaviour == "move_erratic":
            self.move_erratic()
        elif self.behaviour == "flee_from_all":
            self.flee_from_all()
        elif self.behaviour == "flee_from_closest":
            self.flee_from_closest_chaser()
        elif self.behaviour == "flee_from_closest_noise":
            self.flee_from_closest_chaser_with_noise()
        else:
            log.warning("No prey behaviour specified.")
