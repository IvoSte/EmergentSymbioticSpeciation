from .event_manager import QuitEvent, TickEvent

import pygame


class Controller:
    def __init__(self, event_manager, model):
        self.event_manager = event_manager
        self.event_manager.register_listener(self)

        self.model = model

    def notify(self, event):
        if isinstance(event, TickEvent):
            self.parse_input()

    def parse_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.event_manager.post(QuitEvent())

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.event_manager.post(QuitEvent())

                # Place other input controls here.
