import pygame
from ...model.event_manager import TickEvent
from .polygons import (
    FOX_POLYGON,
    RABBIT_FREEK_POLYGON,
)
from ...model.util.polygon_math import (
    move_polyon,
    scale_polygon,
    rotate_polygon,
    flip_polygon,
)
import numpy as np
from random import randint
import math

# class Polygon:
#     base_points: list[tuple[float]]


class AgentView:
    def __init__(self, agent, name, shape, color, scale):
        self.agent = agent
        self.name = name
        self.shape = shape
        self.color = color
        self.scale = scale
        self.trace = []

        # Not sure if these should be here. Perhaps the agent should not make its own drawcalls
        self.name_color = pygame.Color(255, 255, 255)
        self.text_font = pygame.font.SysFont("calibri", 25)

    def _get_polygon(self):
        # scale, rotate, flip, move etc.
        position = self.agent.pos
        polygon = scale_polygon(self.shape, self.scale)
        # angle = math.radians(self.agent.heading)
        # polygon = rotate_polygon(polygon, angle)
        if self.agent.heading % 360 > 90 and self.agent.heading % 360 < 270:
            polygon = flip_polygon(polygon, 0)
        polygon = move_polyon(polygon, position)
        return polygon

    def draw_shape(self, display):
        pygame.draw.polygon(
            display,
            self.color,
            self._get_polygon(),
        )

    def draw_name(self, display):
        display.blit(
            self.text_font.render(self.name, True, self.name_color),
            [self.agent.pos[0] - 15, self.agent.pos[1] - 35],
        )

    def _update_trace(self):
        self.trace += [self.agent.pos]
        if len(self.trace) > 10:
            self.trace.pop(0)

    def draw_trace(self, display):
        pygame.draw.lines(
            display,
            self.color,
            False,
            self.follow_trail[self.agent.name],
            width=2,
        )

    def draw(self, display):
        self.draw_shape(display)
        self.draw_name(display)
        # self.draw_trace(display)


class HunterView(AgentView):
    pass


class Viewer:
    def __init__(self, event_manager, model, config):
        self.event_manager = event_manager
        self.event_manager.register_listener(self)

        self.model = model
        self.display_size = (config["SCREEN_SIZE_X"], config["SCREEN_SIZE_Y"])
        self.display = pygame.display.set_mode(
            (self.display_size[0], self.display_size[1])
        )
        pygame.display.set_caption("Predator-Prey Model")
        self.text_font = pygame.font.SysFont("calibri", 25)

        # self.background_color = pygame.Color(50, 153, 213)
        self.background_color = pygame.Color(84, 201, 66)
        # self.background_color = pygame.Color(144, 200, 51)
        self.text_color = pygame.Color(255, 255, 255)
        self.hunter_color = pygame.Color(222, 0, 0)
        # https://www.schemecolor.com/autumn-fire-color-scheme.php
        self.hunter_type_color = {
            "A": pygame.Color(140, 0, 14),
            "B": pygame.Color(211, 21, 34),
            "C": pygame.Color(255, 68, 23),
            "0": pygame.Color(140, 0, 14),
            "1": pygame.Color(211, 21, 34),
            "2": pygame.Color(255, 68, 23),
            "3": pygame.Color(255, 68, 23),
            "4": pygame.Color(255, 68, 23),
            "default_type": pygame.Color(140, 0, 14),
        }
        self.hunter_type_color.update(
            {n: pygame.Color(255, (68 + n) % 255, 23) for n in range(1000)}
        )
        self.hunter_type_color.update(
            {
                str(n): pygame.Color(
                    randint(0, 255),
                    randint(0, 255),
                    randint(0, 255),
                )
                for n in range(1000)
            }
        )
        # self.hunter_type_color[169] = pygame.Color(211, 21, 34)
        # self.hunter_type_color["169"] = pygame.Color(211, 21, 34)
        # self.hunter_type_color[195] = pygame.Color(255, 68, 23)
        # self.hunter_type_color["195"] = pygame.Color(255, 68, 23)
        # self.hunter_color = pygame.Color(213, 50, 80)
        self.hunter_size = 5
        self.prey_color = pygame.Color(247, 212, 151)
        self.prey_size = 5

        self.helper_overlay = config["HELPER_OVERLAY"]

        self.fox_polygon = flip_polygon(FOX_POLYGON, 1)
        self.rabbit_polygon = flip_polygon(flip_polygon(RABBIT_FREEK_POLYGON, 1), 0)
        self.follow_trail = {}

        self.hunters = [
            AgentView(
                agent=self.model.hunters[0],
                name=self.model.hunters[0].brain.chromosome.chromosome_type,  # "Jaël",
                shape=self.fox_polygon,
                color=pygame.Color(255, 68, 23),
                scale=5,
            ),
            AgentView(
                agent=self.model.hunters[1],
                name=self.model.hunters[
                    0
                ].brain.chromosome.chromosome_type,  # "Thomas",
                shape=self.fox_polygon,
                color=pygame.Color(211, 21, 34),
                scale=6,
            ),
            AgentView(
                agent=self.model.hunters[2],
                name=self.model.hunters[0].brain.chromosome.chromosome_type,  # "Mees",
                shape=self.fox_polygon,
                color=pygame.Color(140, 0, 14),
                scale=2,
            ),
        ]

    def notify(self, event):
        if isinstance(event, TickEvent):
            self.draw()

    def update(self):
        pygame.display.update()

    def clear_screen(self):
        self.display.fill(self.background_color)

    def draw_objects(self):
        # for hunter in self.hunters:
        #     hunter.draw(self.display)

        for hunter in self.model.hunters:
            # pygame.draw.circle(
            #     self.display, self.hunter_color, hunter.pos, self.hunter_size
            # )
            pygame.draw.polygon(
                self.display,
                self.hunter_type_color[hunter.predator_type],
                self.get_agent_polygon(hunter, self.hunter_size, self.fox_polygon),
            )
            #     self.update_follow_trail(hunter)
            #     self.draw_follow_trail(hunter)
            # continue
            self.display.blit(
                self.text_font.render(f"{hunter.predator_type}", True, self.text_color),
                [hunter.pos[0] - 10, hunter.pos[1] - 35],
            )
        for prey in self.model.prey:
            # pygame.draw.circle(self.display, self.prey_color, prey.pos, self.prey_size)
            polygon = self.get_agent_polygon(prey, self.prey_size, self.rabbit_polygon)

            pygame.draw.polygon(
                self.display,
                self.prey_color,
                polygon,
            )

    def get_agent_polygon(self, agent, scale, polygon):
        angle = math.radians(agent.heading)
        position = agent.pos
        polygon = scale_polygon(polygon, scale)
        # polygon = rotate_polygon(polygon, angle)
        if agent.heading % 360 > 90 and agent.heading % 360 < 270:
            polygon = flip_polygon(polygon, 0)
        polygon = move_polyon(polygon, position)
        return polygon
        # scale, rotate, flip, move etc.

    def update_follow_trail(self, agent):
        if agent.name not in self.follow_trail:
            self.follow_trail[agent.name] = [agent.pos]
        self.follow_trail[agent.name] += [agent.pos]
        if len(self.follow_trail[agent.name]) > 10:
            self.follow_trail[agent.name].pop(0)

    def draw_follow_trail(self, agent):
        pygame.draw.lines(
            self.display,
            self.hunter_type_color[agent.predator_type],
            False,
            self.follow_trail[agent.name],
            width=2,
        )

    def draw_helper_overlay(self):
        # for hunter in self.model.hunters:
        #     pygame.draw.line(
        #         self.display,
        #         pygame.Color(255, 50, 0),
        #         hunter.pos,
        #         hunter.target.pos,
        #     )
        text_y_offset = 0.1
        for name, value in self.model.debug_values:
            self.draw_text(name + ": " + value, 0.1, text_y_offset)
            text_y_offset += 0.1

        # return
        text_y_offset = 0.05
        # for hunter in self.model.hunters:
        for hunter in self.model.hunters:
            self.draw_text(
                f"{hunter.predator_type}: {hunter.kills}",
                0.05,
                text_y_offset,
                # color = self.hunter_type_color[hunter.predator_type],
                # pygame.Color(0, 0, 0),
                font=pygame.font.SysFont("calibri", 35),
            )
            text_y_offset += 0.05
        self.draw_text(
            f"tick {self.model.ticks} / {self.model.max_ticks}",
            0.5,
            0.05,
            font=pygame.font.SysFont("calibri", 20),
        )

    def draw_text(self, msg, relative_x, relative_y, color=None, font=None):
        if not isinstance(msg, str):
            msg = f"{msg}"
        if color == None:
            color = self.text_color
        if font == None:
            font = self.text_font

        msg = font.render(msg, True, color)
        self.display.blit(
            msg, [self.display_size[0] * relative_x, self.display_size[1] * relative_y]
        )

    def draw(self):
        self.clear_screen()
        self.draw_objects()
        if self.helper_overlay:
            self.draw_helper_overlay()
        self.update()
