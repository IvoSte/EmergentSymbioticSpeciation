from memory_profiler import profile
from threading import Thread
from typing import Type
from dynaconf import Dynaconf
from multiprocessing import Process, Queue
from speciation.chromosome import Chromosome
from speciation.chromosome_util import save_chromosome_set
from speciation.phylogenetics import PhylogeneticTree

from .sampling import (
    check_all_chromosomes_present,
    check_compositions_completeness,
    align_compositions_with_chromosome_population_types,
    compositions_generation_overview,
    generate_chromosome_combinations,
    generate_chromosome_sets_from_compositions,
    check_chromosome_sets,
)
from shared_components.model_data import ModelData, ModelRunData, ModelGenerationData
from shared_components.model_interface import Model
from shared_components.parameters import Parameters
from speciation.evolution_manager import EvolutionManager


class MultiRunner:
    def __init__(
        self,
        model: Type[Model],
        config: Dynaconf,
        evolution_manager: EvolutionManager = None,
    ):
        self.model = model
        self.evolution_manager = evolution_manager
        self.run_benchmark = config["RUN_BENCHMARK"]
        self.benchmark_interval = config["BENCHMARK_INTERVAL"]
        self.multi_threading = config["MULTITHREADING"]
        self.sampling_method = config["CHROMOSOME_SAMPLING"]
        self.save_chromosomes = config["SAVE_CHROMOSOMES"]

        if config["RUN_WITH_VIEWER"]:
            import pygame

            pygame.init()

    def run_model_x_times_per_parameter_set(
        self, runs_per_set: int, parameter_set: list[Parameters]
    ) -> list[ModelRunData]:
        data = []
        for parameter_idx, parameters in enumerate(parameter_set):
            if self.multi_threading:
                data += self.run_model_x_times_with_parameters_parallel(
                    runs_per_set, parameters, parameter_idx
                )
            else:
                data += self.run_model_x_times_with_parameters(
                    runs_per_set, parameters, parameter_idx
                )
        return data

    def run_model_x_times_with_parameters(
        self, runs_per_set: int, parameters: Parameters, parameter_idx: int
    ) -> list[ModelRunData]:
        data = []
        for run_idx in range(runs_per_set):
            # print(
            #     f"Run {run_idx} for parameter set {parameter_idx}, total run {(parameter_idx * runs_per_set) + run_idx}",
            #     end="\r",
            # )
            # print(parameters)
            model = self.model(parameters)
            model.run()
            data.append(
                ModelRunData(
                    total_run_number=(parameter_idx * runs_per_set) + run_idx,
                    parameter_run_number=run_idx,
                    parameter_id=parameter_idx,
                    model_data=model.get_data(),
                )
            )
        return data

    def run_model_x_times_with_parameters_parallel(
        self, runs_per_set, parameters: dict, parameter_idx: int
    ) -> list[ModelRunData]:
        data = []
        queue = Queue()
        models = [self.model(parameters) for _ in range(runs_per_set)]
        run_threads = [Thread(target=model.run) for model in models]
        get_data_threads = [
            Thread(target=(lambda model: queue.put(model.get_data()))(model))
            for model in models
        ]

        # LATER TODO This can be made into a threadpool.

        for thread in run_threads:
            thread.start()
        for thread in run_threads:
            thread.join()

        for thread in get_data_threads:
            thread.start()
        for thread in get_data_threads:
            thread.join()

        model_data_list = []
        while not queue.empty():
            model_data_list.append(queue.get())

        for run_idx, model_data in enumerate(model_data_list):
            data.append(
                ModelRunData(
                    total_run_number=(parameter_idx * runs_per_set) + run_idx,
                    parameter_run_number=run_idx,
                    parameter_id=parameter_idx,
                    model_data=model_data,
                )
            )

        return data

    def run_model_benchmark_for_parameters(self, parameter_set: list[Parameters]):
        results = []
        try:
            for parameter in parameter_set:
                model = self.model(parameter)
                results.append(model.run_benchmark())
        except Exception as e:
            print(
                f"Error in running model benchmarks, possibly the model does not have a run_benchmark() defined"
            )
            print(e)
            return None
        return sum(results) / len(results)

    def run_model_benchmark_for_parameters_verbose(
        self, parameter_set: list[Parameters]
    ):
        results = {}
        try:
            for parameter in parameter_set:
                type_combination = str(
                    sorted(
                        [
                            chromosome.chromosome_type
                            for chromosome in parameter.agent_chromosomes
                        ]
                    )
                )
                if type_combination not in results:
                    results[type_combination] = []
                model = self.model(parameter)
                benchmark_score = model.run_benchmark()
                results[type_combination].append(benchmark_score)
                # TEST CODE REMOVE AFTER #
                if self.save_chromosomes and benchmark_score == 9.0:
                    save_chromosome_set(parameter.agent_chromosomes)
                    print(f"saved best chromosome set with score {benchmark_score}")
                # /TEST CODE REMOVE AFTER #

        except Exception as e:
            print(
                f"Error in running model benchmarks, possibly the model does not have a run_benchmark() defined"
            )
            print(e)
            return None

        # A list of all chromosome types in all parameters in the parameter set
        all_types = [
            chromosome.chromosome_type
            for parameter in parameter_set
            for chromosome in parameter.agent_chromosomes
        ]
        average_type_counts = {
            type_: all_types.count(type_) / len(parameter_set)
            for type_ in set(all_types)
        }
        # sort the results by average score
        average_combination = {
            k: v
            for k, v in sorted(
                average_type_counts.items(), key=lambda item: item[1], reverse=True
            )
        }

        all_results = [score for scores in results.values() for score in scores]
        average_of_all = sum(all_results) / len(all_results)
        average_per_combination = {
            type_combination: sum(results[type_combination])
            / len(results[type_combination])
            for type_combination in results
        }
        best_combination = max(average_per_combination, key=average_per_combination.get)
        best_combination_score = average_per_combination[best_combination]
        best_combination_count = len(results[best_combination])
        worst_combination = min(
            average_per_combination, key=average_per_combination.get
        )
        worst_combination_score = average_per_combination[worst_combination]
        result = {
            "average": average_of_all,
            "average_combination": average_combination,
            "best_combination": best_combination,
            "best_combination_score": best_combination_score,
            "best_combination_count": best_combination_count,
            "worst_combination": worst_combination,
            "worst_combination_score": worst_combination_score,
        }
        return result

    def run_model_benchmark(self, parameter_set: list[Parameters], generation: int):
        # If necessary, run the benchmark for the current generation and collect the results
        if self.run_benchmark and (generation) % self.benchmark_interval == 0:
            benchmark_results = self.run_model_benchmark_for_parameters_verbose(
                parameter_set
            )
            # Leave this print here
            print(
                f"Benchmark for generation {generation}: \n\taverage {benchmark_results['average']:.3f} - {benchmark_results['average_combination']}\n\tbest {benchmark_results['best_combination_score']:.3f} - {benchmark_results['best_combination']}\n\tworst {benchmark_results['worst_combination_score']:.3f} - {benchmark_results['worst_combination']}"
            )
        else:
            benchmark_results = {
                "average": None,
                "best_combination": None,
                "best_combination_score": None,
                "best_combination_count": None,
                "worst_combination": None,
                "worst_combination_score": None,
            }
        return benchmark_results

    def run_model_generation(
        self, parameter_set: list[Parameters], runs_per_set: int
    ) -> list[ModelRunData]:
        generation_data = self.run_model_x_times_per_parameter_set(
            runs_per_set, parameter_set
        )
        return generation_data

    def evolve_generation_chromosomes(
        self, generation_data: list[ModelRunData], generation: int
    ) -> tuple[list[Chromosome], dict]:
        new_chromosomes, generation_info = self.evolution_manager.evolve_generation(
            generation_data=generation_data, generation=generation
        )
        return new_chromosomes, generation_info

    # TODO this function could/should possibly be refactored to the evolutionary manager.
    def generate_new_chromosome_combinations(
        self,
        new_chromosomes: list[Chromosome],
        generation_data: list[ModelRunData],
        n_combinations: int,
        composition_size: int,
    ) -> list[list[Chromosome]]:
        if self.sampling_method == "evolutionary":
            chromosome_combinations = (
                self.evolution_manager.generate_new_chromosome_combinations(
                    new_chromosomes, generation_data
                )
            )
            # new_chromosome_types = self.evolution_manager.get_chromosome_types(
            #     new_chromosomes
            # )
            # chromosome_commpositions = (
            #     self.evolution_manager.evolve_chromosome_compositions(
            #         generation_data=generation_data,
            #         allowed_gene_pool=new_chromosome_types,
            #     )
            # )
            # chromosome_commpositions = (
            #     align_compositions_with_chromosome_population_types(
            #         chromosome_commpositions, new_chromosomes
            #     )
            # )
            # # for composition in chromosome_commpositions:
            # #     print(composition)
            # chromosome_combinations = generate_chromosome_sets_from_compositions(
            #     chromosome_commpositions, new_chromosomes
            # )
            # TEST
            # compositions_generation_overview(chromosome_commpositions, new_chromosomes)
            # check_chromosome_sets(chromosome_combinations)
            # check_compositions_completeness(
            #     chromosome_commpositions, new_chromosome_types
            # )
            # /TEST
        else:
            chromosome_combinations = generate_chromosome_combinations(
                chromosomes=new_chromosomes,
                n_combinations=n_combinations,
                combination_size=composition_size,
                method=self.sampling_method,
            )
        # TEST
        # check_all_chromosomes_present(new_chromosomes, chromosome_combinations)
        # /TEST
        return chromosome_combinations

    def update_parameter_set_with_chromosome_combinations(
        self,
        parameter_set: list[Parameters],
        chromosome_combinations: list[list[Chromosome]],
    ) -> list[Parameters]:
        assert len(parameter_set) == len(
            chromosome_combinations
        ), f"{len(parameter_set) = } {len(chromosome_combinations) = }"
        for idx, parameter in enumerate(parameter_set):
            parameter.agent_chromosomes = chromosome_combinations[idx]
        return parameter_set

    def run_species_detection_tests(
        self, new_chromosomes: list[Chromosome], generation: int
    ):
        self.tree.add_chromosme_generation(new_chromosomes)
        if generation > 1 and generation % 10 == 0:
            pass
            # print(f"Generation {generation} species detection test.")
            # self.tree.species_detection_consistency_test(generation - 1)
            # self.tree.calculate_distances_to_prototypes(generation)
            # self.tree.calculate_distances_between_prototypes(generation)
            # self.tree.current_species_distribution(generation - 1)

            # self.evolution_manager.species_detector.species_detection_test(
            #     new_chromosomes
            # )

    # Name can be changed, this one is specifically for evolutionary process
    # This function does too much, but it's hard to refactor without spending too much time.
    def run_model_for_generations(
        self, generations: int, runs_per_set: int, parameter_set: list[Parameters]
    ) -> list[ModelGenerationData]:
        data = []

        self.tree = PhylogeneticTree(self.evolution_manager.species_detector)

        for generation in range(generations):
            print(f"Generation {generation}", end="\r")

            # Run the model for the current parameter set and collect the results
            generation_data = self.run_model_generation(parameter_set, runs_per_set)

            # If necessary, run the benchmark for the current generation and collect the results
            benchmark_results = self.run_model_benchmark(parameter_set, generation)

            # Evolve the current generation and collect the new chromosomes and generation info
            new_chromosomes, generation_info = self.evolve_generation_chromosomes(
                generation_data, generation
            )
            # Run species detection test -- TEMPorary testing function
            # self.run_species_detection_tests(new_chromosomes, generation)

            # Code only used for very few experiments so it's not properly integrated but does work. ###
            extinction_event = False
            if extinction_event and generation == 250:
                new_chromosomes = self.evolution_manager.extinction_event(
                    new_chromosomes
                )
            ###

            # Create new chromosome combinations
            chromosome_combinations = self.generate_new_chromosome_combinations(
                new_chromosomes=new_chromosomes,
                generation_data=generation_data,
                n_combinations=len(parameter_set),
                composition_size=len(parameter_set[0].agent_chromosomes),
            )

            # Place the combinations in the parameter set
            parameter_set = self.update_parameter_set_with_chromosome_combinations(
                parameter_set, chromosome_combinations
            )

            data.append(
                ModelGenerationData(
                    generation_number=generation,
                    best_fitness_per_species=generation_info[
                        "best_fitness_per_species"
                    ],
                    average_fitness_per_species=generation_info[
                        "average_fitness_per_species"
                    ],
                    benchmark_score=benchmark_results["average"],
                    benchmark_best_combination=benchmark_results["best_combination"],
                    benchmark_best_combination_score=benchmark_results[
                        "best_combination_score"
                    ],
                    benchmark_best_combination_count=benchmark_results[
                        "best_combination_count"
                    ],
                    benchmark_worst_combination=benchmark_results["worst_combination"],
                    benchmark_worst_combination_score=benchmark_results[
                        "worst_combination_score"
                    ],
                    generation_model_run_data=generation_data,
                )
            )
        return data
