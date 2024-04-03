from dataclasses import dataclass
from weakref import WeakKeyDictionary
from shared_components.logger import log


@dataclass
class Event:
    name: str = "Generic Event"


@dataclass
class TickEvent(Event):
    name: str = "Tick Event"


@dataclass
class QuitEvent(Event):
    name: str = "Quit Event"


class EventManager:
    def __init__(self):
        self.listeners = WeakKeyDictionary()

    def register_listener(self, listener):
        self.listeners[listener] = 1

    def unregister_listener(self, listener):
        if listener in self.listeners:
            del self.listeners[listener]

    def post(self, event):
        log.debug(f"Event {event} posted.")
        for listener in self.listeners:
            listener.notify(event)
