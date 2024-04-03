import sys

# sys.path.insert(0, "../../")

from .model.predator_prey import PredatorPrey
from .model.parameters import PPParameters
from .model.event_manager import EventManager
from .model.controller import Controller
from .model.viewer.viewer import Viewer


class PredatorPreyViewer:
    def __init__(self, config):
        assert (
            config["RUN_WITH_VIEWER"] == True
        ), "Viewer must be enabled for viewer run"
        self.config = config
        self.event_manager = EventManager()
        self.parameters = PPParameters.from_config(self.config)
        self.model = PredatorPrey(
            parameters=self.parameters,
            event_manager=self.event_manager,
            run_with_viewer=True,
            frames_per_second=self.config["FRAMES_PER_SECOND"],
        )
        self.controller = Controller(self.event_manager, self.model)
        self.viewer = Viewer(self.event_manager, self.model, self.config)

    def view_run(self):
        self.model.run()

    def view_benchmark(self, chromosomes: list = None):
        if chromosomes:
            self.model.set_hunters(chromosomes)
        print(f"Successes: {self.model.run_benchmark()}")
