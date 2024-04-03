# models/model_factory.py

from typing import Type
from enum import Enum

from .toxin.model.toxin_model import ToxinModel
from .predator_prey.model.predator_prey import PredatorPrey
from .function_optimization.model.function_model import BenchmarkFunctionModel
from shared_components.model_interface import Model
from shared_components.model_analysis import ModelAnalysis
from shared_components.model_visualizer import ModelVisualizer
from shared_components.parameters import Parameters

from .predator_prey.model.parameters import PPParameters
from .toxin.model.parameters import ToxinParameters
from .function_optimization.model.parameters import BenchmarkFunctionParameters
from .predator_prey.visualization.predator_prey_visualizer import PredatorPreyVisualizer
from .toxin.visualization.toxin_visualizer import ToxinVisualizer
from .function_optimization.visualization.function_optimization_visualizer import (
    BenchmarkFunctionVisualizer,
)
from .predator_prey.visualization.predator_prey_analysis import (
    PredatorPreyAnalysis,
)
from .toxin.visualization.toxin_analysis import ToxinAnalysis
from .function_optimization.visualization.function_optimization_analysis import (
    FunctionOptimizationAnalysis,
)


class ModelType(Enum):
    PREDATOR_PREY = "predator_prey"
    TOXIN = "toxin"
    FUNCTION_OPTIMIZATION = "function_optimization"


class ModelFactory:
    def __init__(self, model_type: ModelType):
        self.model_type = model_type

    @property
    def model(self) -> Type[Model]:
        if self.model_type == ModelType.PREDATOR_PREY:
            return PredatorPrey
        elif self.model_type == ModelType.TOXIN:
            return ToxinModel
        elif self.model_type == ModelType.FUNCTION_OPTIMIZATION:
            return BenchmarkFunctionModel
        else:
            raise Exception("Model type not recognized")

    @property
    def parameters(self) -> Type[Parameters]:
        if self.model_type == ModelType.PREDATOR_PREY:
            return PPParameters
        elif self.model_type == ModelType.TOXIN:
            return ToxinParameters
        elif self.model_type == ModelType.FUNCTION_OPTIMIZATION:
            return BenchmarkFunctionParameters
        else:
            raise Exception("Model type not recognized")

    @property
    def visualizer(self) -> Type[ModelVisualizer]:
        if self.model_type == ModelType.PREDATOR_PREY:
            return PredatorPreyVisualizer
        elif self.model_type == ModelType.TOXIN:
            return ToxinVisualizer
        elif self.model_type == ModelType.FUNCTION_OPTIMIZATION:
            return BenchmarkFunctionVisualizer
        else:
            raise Exception("Model type not recognized")

    @property
    def data_analysis(self) -> Type[ModelAnalysis]:
        if self.model_type == ModelType.PREDATOR_PREY:
            return PredatorPreyAnalysis
        elif self.model_type == ModelType.TOXIN:
            return ToxinAnalysis
        elif self.model_type == ModelType.FUNCTION_OPTIMIZATION:
            return FunctionOptimizationAnalysis
        else:
            raise Exception("Model type not recognized")

    # @staticmethod
    # def create_model(model_name: str, **model_params) -> Model:
    #     if model_name == "toxin":
    #         return ToxinModel(**model_params)

    #     elif model_name == "predator_prey":
    #         return PredatorPrey(**model_params)

    #     else:
    #         raise ValueError(f"Invalid model name '{model_name}'")
