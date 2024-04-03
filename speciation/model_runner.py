import time
from models.predator_prey.predator_prey_viewer import PredatorPreyViewer
from shared_components.logger import log
from shared_components.model_data import ModelSuperRunData
from shared_components.parameters import Parameters
from speciation.chromosome_util import load_chromosome_set, save_chromosome_set
from speciation.model_plot_generator import ModelPlotGenerator

# from speciation.species_graph import SpeciesGraph
from .evolution_machine import (
    CrossoverType,
    EvolutionMachine,
    GeneDataType,
    SelectionType,
    MateSelectionType,
)
from .evolution_manager import EvolutionManager
from .hyperparameters import HyperParameters
from .multi_runner import MultiRunner
from models.model_factory import ModelFactory, ModelType


class ModelRunner:
    def __init__(self, config):
        self.config = config
        self.model_type = ModelType(config["MODEL"])
        self.model_factory = ModelFactory(self.model_type)
        evolution_machine = EvolutionMachine(
            reproduce_fraction=config["AGENT"]["REPRODUCE_FRACTION"],
            cull_fraction=config["AGENT"]["CULL_FRACTION"],
            mutation_probability=config["AGENT"]["MUTATION_PROBABILITY"],
            mutation_range=config["AGENT"]["MUTATION_RANGE"],
            expected_population_size=config["SUBPOPULATION_SIZE"],
            selection_type=SelectionType(config["AGENT"]["SELECTION_TYPE"]),
            mate_selection=MateSelectionType(config["AGENT"]["MATE_SELECTION_TYPE"]),
            mate_selection_sample_size=config["AGENT"]["MATE_SELECTION_SAMPLE_SIZE"],
            allow_self_mating=config["AGENT"]["ALLOW_SELF_MATING"],
            gene_data_type=GeneDataType(config["AGENT"]["GENE_DATA_TYPE"]),
            knockout_mutation=config["AGENT"]["KNOCKOUT_MUTATION"],
            knockout_mutation_probability=config["AGENT"][
                "KNOCKOUT_MUTATION_PROBABILITY"
            ],
        )
        self.evolution_manager = EvolutionManager(evolution_machine, self.config)
        if config["CHROMOSOME_SAMPLING"] == "evolutionary":
            composition_evolution_machine = EvolutionMachine(
                reproduce_fraction=config["COMPOSITION"]["REPRODUCE_FRACTION"],
                cull_fraction=config["COMPOSITION"]["CULL_FRACTION"],
                mutation_probability=config["COMPOSITION"]["MUTATION_PROBABILITY"],
                mutation_range=config["COMPOSITION"]["MUTATION_RANGE"],
                selection_type=SelectionType(config["COMPOSITION"]["SELECTION_TYPE"]),
                mate_selection=MateSelectionType(
                    config["COMPOSITION"]["MATE_SELECTION_TYPE"]
                ),
                mate_selection_sample_size=config["COMPOSITION"][
                    "MATE_SELECTION_SAMPLE_SIZE"
                ],
                allow_self_mating=config["COMPOSITION"]["ALLOW_SELF_MATING"],
                gene_data_type=GeneDataType.NOMINAL,
                chromosome_resizing=config["COMPOSITION"]["CHROMOSOME_RESIZING"],
                chromosome_resizing_probability=config["COMPOSITION"][
                    "CHROMOSOME_RESIZING_PROBABILITY"
                ],
                crossover_type=CrossoverType.UNIFORM_VARIABLE_LENGTH,
            )
            self.evolution_manager.add_composition_evolution_machine(
                composition_evolution_machine
            )

        self.hyperparameters = HyperParameters(
            self.model_factory.parameters, self.config
        )
        self.multi_runner = MultiRunner(
            model=self.model_factory.model,
            config=self.config,
            evolution_manager=self.evolution_manager,
        )

    def _get_initial_parameter_set(self) -> list[Parameters]:
        if (
            self.config["MODEL"] == "function_optimization"
        ):  # For the optimization functions model, we need initial gene value boundaries
            gene_value_boundaries = self.config[self.config["FUNCTION_NAME"]][
                "GENE_VALUE_BOUNDARIES"
            ]
        else:
            gene_value_boundaries = None
        initial_chromosomes = self.evolution_manager.generate_chromosomes(
            chromosome_length=self.config["CHROMOSOME_LENGTH"],
            chromosome_types=[i for i in range(self.config["N_SUBPOPULATIONS"])],
            n_chromosomes_per_type=self.config["SUBPOPULATION_SIZE"],
            binary=self.config["AGENT"]["GENE_DATA_TYPE"] == "binary",
            gene_value_boundaries=gene_value_boundaries,
            initial_gene_knockout=self.config["AGENT"]["INITIAL_GENE_KNOCKOUT"],
        )
        if self.config["EVOLUTION_TYPE"] == "emergent_subpopulations":
            initial_chromosomes = self.evolution_manager.assign_chromosome_types(
                initial_chromosomes
            )
        parameter_set = self.hyperparameters.generate_parameter_set_from_chromosomes(
            chromosomes=initial_chromosomes,
            n_parameters=self.config["N_COMPOSITIONS"],
            chromosomes_per_set=self.config["N_AGENTS"],
            sampling_method=self.config["CHROMOSOME_SAMPLING"],
        )
        return parameter_set

    def run(self, verbose=False) -> list[ModelSuperRunData]:
        if verbose:
            return self.run_verbose()
        else:
            return self.run_silent()

    def run_silent(self) -> list[ModelSuperRunData]:
        data = []
        for run_idx in range(self.config["N_RUNS"]):
            self.multi_runner.evolution_manager.species_detector.species_tracker.reset()
            data.append(
                ModelSuperRunData(
                    super_run_number=run_idx,
                    generation_data=self.multi_runner.run_model_for_generations(
                        self.config["N_GENERATIONS"],
                        self.config["RUNS_PER_SET"],
                        self._get_initial_parameter_set(),
                    ),
                )
            )
        return data

    def run_verbose(self) -> list[ModelSuperRunData]:
        start = time.perf_counter()
        log.info(f"Running {self.model_type.value} model {self.config['N_RUNS']} times")
        data = []
        for run_idx in range(self.config["N_RUNS"]):
            self.multi_runner.evolution_manager.species_detector.species_tracker.reset()
            data.append(
                ModelSuperRunData(
                    super_run_number=run_idx,
                    generation_data=self.multi_runner.run_model_for_generations(
                        self.config["N_GENERATIONS"],
                        self.config["RUNS_PER_SET"],
                        self._get_initial_parameter_set(),
                    ),
                )
            )
        end = time.perf_counter()
        print(f"Finished in {round(end-start, 2)}second(s)")
        return data

    def view_chromosome_benchmark_behaviour(self, loop=False):
        if self.model_type != ModelType.PREDATOR_PREY:
            log.warning("Only the Predator Prey model has a viewer.")
            return
        chromosomes = load_chromosome_set()
        if loop:
            while True:
                PredatorPreyViewer(self.config).view_benchmark(chromosomes)
        else:
            PredatorPreyViewer(self.config).view_benchmark(chromosomes)

    def visualize_results(self, data):
        start = time.perf_counter()
        model_visualizer = ModelPlotGenerator(self.config)

        v = model_visualizer.visualize_results(data, separate_images=False)
        # model_visualizer.visualize_species_graph(
        #     self.evolution_manager.species_detector.species_tracker.species
        # )
        print(
            f"Finished loading data in : {round(time.perf_counter()-start, 2)}second(s)"
        )

        v.show()
