from .logger import log


class Model:
    def __init__(self, parameters):
        self.parameters = parameters

    def run(self):
        log.error("Trying to run Model interface.")

    def get_data(self):
        log.error("Trying to get data from Model interface.")
