from .logger import log
from dataclasses import dataclass


@dataclass
class Parameters:
    def from_config(config):
        log.error("Trying to instantiate Parameters interface as concrete object.")
        raise NotImplementedError
