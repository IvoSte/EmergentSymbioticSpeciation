import random
from uuid import uuid4

from dynaconf import Dynaconf
from shared_components.model_data import ModelRunData
from shared_components.logger import log
from dataclasses import dataclass
from speciation.sampling import (
    align_compositions_with_chromosome_population_types,
    generate_chromosome_sets_from_compositions,
)

from speciation.species_tracking import SpeciesTracker
from .chromosome import Chromosome
from .species_detection import SpeciesDetector
from .evolution_machine import EvolutionMachine


@dataclass
class ChromosomeMetaData:
    chromosome_id: str
    chromosome_type: str

    genes: list[float]

    individual_fitness_scores: list[float]
    collective_fitness_scores: list[float]
    fitness_scores: list[float]

    individual_fitness: float
    collective_fitness: float
    fitness: float


class EvolutionManager:
    # This class is the interface between a model run, and all extra model evolutionary processes.
    # In its simplest form, it receives model (generation) run data, and returns a set of parameters (including chromosomes)
    # for the model to run in the next generation
    def __init__(
        self,
        evolution_machine: EvolutionMachine,
        config: Dynaconf,
    ):
        self.evolution_machine = evolution_machine
        self.collective_fitness_weight = config["COLLECTIVE_FITNESS_WEIGHT"]
        self.detect_species_interval = config["DETECT_SPECIES_INTERVAL"]
        self.detect_species_species_with_previous_generation = config[
            "DETECT_SPECIES_WITH_PREVIOUS_GENERATION"
        ]
        self.evolution_type = config["EVOLUTION_TYPE"]
        self.scale_species_fitness_with_composition_representation = config[
            "SCALE_FITNESS_WITH_REPRESENTATION"
        ]
        self.align_compositions_with_population = config[
            "ALIGN_COMPOSITIONS_WITH_POPULATION"
        ]

        self.species_detector = SpeciesDetector(
            method=config["DETECT_SPECIES_METHOD"],
            calibrate_clusters_delta_range=config["DETECT_SPECIES_DELTA_RANGE"],
            eps=config["DETECT_SPECIES_EPS"],
            min_samples=config["DETECT_SPECIES_MIN_SAMPLES"],
            species_tracking=config["SPECIES_TRACKING"],
            species_tracker=SpeciesTracker(
                search_depth=config["SPECIES_TRACKING_SEARCH_DEPTH"],
                distance_threshold=config["SPECIES_TRACKING_DISTANCE_THRESHOLD"],
            ),
            prototype_method=config["SPECIES_TRACKING_PROTOTYPE_METHOD"],
        )

    def add_composition_evolution_machine(
        self, composition_evolution_machine: EvolutionMachine
    ):
        self.composition_evolution_machine = composition_evolution_machine

    def evolve_generation(
        self,
        generation_data: list[ModelRunData],
        generation: int,
    ) -> tuple[list[Chromosome], dict]:
        chromosome_fitness_per_species = (
            self._get_chromosome_fitness_per_species_from_generation_data(
                generation_data
            )
        )

        if self.scale_species_fitness_with_composition_representation:
            chromosome_fitness_per_species = (
                self._scale_species_fitness_with_composition_representation(
                    chromosome_fitness_per_species, generation_data
                )
            )

        if self.evolution_type == "forced_subpopulations":
            # Evolve per species / chromosome_type (forced subpopulations)
            new_chromosomes = self._evolve_chromosomes_per_species(
                chromosome_fitness_per_species
            )
        elif self.evolution_type == "emergent_subpopulations":
            # Evolve all (free subpopulations)
            detect_species = generation % self.detect_species_interval == 0
            new_chromosomes = self._evolve_chromosomes_full_population(
                chromosome_fitness_per_species, detect_species
            )
        generation_info = self._get_generation_info(chromosome_fitness_per_species)
        # elif self.evolution_type == "emergent_subpopulations":
        return new_chromosomes, generation_info

    def generate_chromosomes(
        self,
        chromosome_length,
        chromosome_types,
        n_chromosomes_per_type,
        binary=False,
        gene_value_boundaries: tuple = None,
        initial_gene_knockout: int = None,
    ) -> list[Chromosome]:
        chromosomes = []
        for chromosome_type in chromosome_types:
            for _ in range(n_chromosomes_per_type):
                chromosome = Chromosome(
                    chromosome_id=uuid4(), chromosome_type=chromosome_type
                )
                if binary:
                    chromosome.init_random_binary(chromosome_length)
                elif gene_value_boundaries != None:
                    chromosome.init_random_with_range(
                        chromosome_length, *gene_value_boundaries
                    )
                else:
                    chromosome.init_random(chromosome_length)
                if initial_gene_knockout:
                    knockout_n_genes = (
                        min(initial_gene_knockout, chromosome_length - 1)
                        if initial_gene_knockout != -1
                        else random.randint(0, chromosome_length - 1)
                    )
                    chromosome.knockout_genes(knockout_n_genes)
                chromosomes.append(chromosome)
        return chromosomes

    def assign_chromosome_types(
        self, chromosomes: list[Chromosome]
    ) -> list[Chromosome]:
        genes = [chromosome.genes for chromosome in chromosomes]
        detected_types = self.species_detector.get_quasispecies(genes)
        for chromosome, chromosome_type in zip(chromosomes, detected_types):
            chromosome.chromosome_type = chromosome_type
        return chromosomes

    def get_chromosome_types(self, chromosomes: list[Chromosome]) -> list:
        types = set()
        for chromosome in chromosomes:
            types.add(chromosome.chromosome_type)
        return list(types)

    def _evolve_chromosomes_per_species(self, chromosome_fitness_per_species):
        sorted_fitness = {}
        new_chromosomes = []

        # Evolve per species / chromosome_type (forced subpopulations)
        for (
            chromosome_type,
            chromosome_species_fitness,
        ) in chromosome_fitness_per_species.items():
            # sort on fitness
            sorted_chromosome_fitness = sorted(
                chromosome_species_fitness, key=lambda kv: kv[1], reverse=True
            )
            sorted_chromosomes, sorted_fitness[chromosome_type] = map(
                list, zip(*sorted_chromosome_fitness)
            )
            # get new genes
            new_genes = self.evolution_machine.evolution_step(
                sorted_chromosomes, sorted_fitness[chromosome_type]
            )
            # generate chromosomes from genes
            new_chromosomes += [
                Chromosome(genes=genes, chromosome_type=chromosome_type)
                for genes in new_genes
            ]
        return new_chromosomes

    def _evolve_chromosomes_full_population(
        self, chromosome_fitness_per_species, detect_species
    ):
        # chromosome_fitness_per_species is a dict with chromosome_type as key, and a list of tuples (genes, fitness) as value

        new_chromosomes = []
        num_species = len(chromosome_fitness_per_species)

        # Evolve all (free subpopulations)
        all_chromosome_fitness = []
        for chromosome_species_fitness in chromosome_fitness_per_species.values():
            all_chromosome_fitness += chromosome_species_fitness

        # sort on fitness
        sorted_chromosome_fitness = sorted(
            all_chromosome_fitness, key=lambda kv: kv[1], reverse=True
        )
        sorted_chromosomes, chromosome_fitness = map(
            list, zip(*sorted_chromosome_fitness)
        )
        # get new genes
        new_genes = self.evolution_machine.evolution_step(
            sorted_chromosomes, chromosome_fitness
        )

        # determine chromosome type
        if self.detect_species_species_with_previous_generation:
            new_genes_species = self.species_detector.get_quasispecies(
                genes=new_genes + sorted_chromosomes,
                num_species=num_species,
                detect_species=detect_species,
            )[: len(new_genes)]
        else:
            new_genes_species = self.species_detector.get_quasispecies(
                genes=new_genes,
                num_species=num_species,
                detect_species=detect_species,
            )

        # generate chromosomes from genes
        for genes, species in zip(new_genes, new_genes_species):
            new_chromosomes.append(Chromosome(genes=genes, chromosome_type=species))

        return new_chromosomes

    def _get_generation_info(self, chromosome_fitness_per_species):
        fitness_per_species = {
            chromosome_type: [
                chromosome_fitness_per_species[chromosome_type][idx][1]
                for idx in range(len(chromosome_fitness_per_species[chromosome_type]))
            ]
            for chromosome_type in chromosome_fitness_per_species
        }
        best_fitness_per_species = {
            chromosome_type: max(fitness_per_species[chromosome_type])
            for chromosome_type in fitness_per_species
        }
        average_fitness_per_species = {
            chromosome_type: sum(fitness_per_species[chromosome_type])
            / len(fitness_per_species[chromosome_type])
            for chromosome_type in fitness_per_species
        }
        return {
            "best_fitness_per_species": best_fitness_per_species,
            "average_fitness_per_species": average_fitness_per_species,
        }

    def _get_chromosome_fitness_per_species_from_generation_data(
        self, generation_data: list[ModelRunData]
    ) -> dict[list[tuple]]:
        # The aim of this function is to parse all ModelRunData produced by a single generation,
        # and return a list with the unique chromosomes and their average fitness, by species.
        # chromosome_fitness[type_] = list[tuple(chromosome: list[float], avg_fitness: float), ..]

        chromosome_metadata_table = self._get_chromosome_metadata_table(generation_data)

        chromosome_fitness = {}

        for chromosome_metadata in chromosome_metadata_table.values():
            if chromosome_metadata.chromosome_type not in chromosome_fitness:
                chromosome_fitness[chromosome_metadata.chromosome_type] = []

            chromosome_fitness[chromosome_metadata.chromosome_type].append(
                (
                    chromosome_metadata.genes,
                    chromosome_metadata.fitness,
                )
            )

        return chromosome_fitness

    def _get_chromosome_metadata_table(self, generation_data: list[ModelRunData]):
        chromosome_ids = {
            single_agent_data["chromosome_id"]
            for model_run_data in generation_data
            for single_agent_data in model_run_data.model_data.agent_data
        }

        chromosome_metadata_table = {
            chromosome_id: ChromosomeMetaData(
                chromosome_id=chromosome_id,
                individual_fitness_scores=[],
                collective_fitness_scores=[],
                fitness_scores=[],
                chromosome_type="",
                genes=[],
                individual_fitness=0.0,
                collective_fitness=0.0,
                fitness=0.0,
            )
            for chromosome_id in chromosome_ids
        }
        # get values per run
        for model_run_data in generation_data:
            for single_agent_data in model_run_data.model_data.agent_data:
                # NOTE optimization possible, stuff gets overwritten. Make it to an object could also help, but dictionar
                chromosome_metadata_table[
                    single_agent_data["chromosome_id"]
                ].genes = single_agent_data["chromosome"]

                chromosome_metadata_table[
                    single_agent_data["chromosome_id"]
                ].chromosome_type = single_agent_data["chromosome_type"]

                chromosome_metadata_table[
                    single_agent_data["chromosome_id"]
                ].individual_fitness_scores.append(
                    single_agent_data["individual_fitness"]
                )

                chromosome_metadata_table[
                    single_agent_data["chromosome_id"]
                ].collective_fitness_scores.append(
                    single_agent_data["collective_fitness"]
                )

                chromosome_metadata_table[
                    single_agent_data["chromosome_id"]
                ].fitness_scores.append(single_agent_data["combined_fitness"])

        # calculate averages
        for chromosome_metadata in chromosome_metadata_table.values():
            chromosome_metadata.individual_fitness = sum(
                chromosome_metadata.individual_fitness_scores
            ) / len(chromosome_metadata.individual_fitness_scores)

            chromosome_metadata.collective_fitness = sum(
                chromosome_metadata.collective_fitness_scores
            ) / len(chromosome_metadata.collective_fitness_scores)

            chromosome_metadata.fitness = sum(chromosome_metadata.fitness_scores) / len(
                chromosome_metadata.fitness_scores
            )

        return chromosome_metadata_table

    def _get_compositions_fitness(self, generation_data: list[ModelRunData]):
        # This function can be prettier / more readable, but this is faster.
        compositions = {}
        for model_run in generation_data:
            parameter_id = model_run.parameter_id

            if parameter_id not in compositions:
                compositions[parameter_id] = {
                    "composition": [],
                    "fitness": [],
                }
            for agent_data in model_run.model_data.agent_data:
                # Add the type to the composition only if it is the first run of the parameter
                if model_run.parameter_run_number == 0:
                    compositions[parameter_id]["composition"].append(
                        agent_data["chromosome_type"]
                    )
                # The fitness is added multiple times per composition, but since it is the same for all, the average stays the same.
                # Possible optimization: only add the fitness once per composition.
                compositions[parameter_id]["fitness"].append(
                    agent_data["collective_fitness"]
                )
            # Sort the composition combination, to collapse permutations of the same composition.
            compositions[parameter_id]["composition"].sort()

        composition_chromosomes = [
            composition["composition"] for composition in compositions.values()
        ]
        composition_fitness = [
            sum(composition["fitness"]) / len(composition["fitness"])
            for composition in compositions.values()
        ]
        return composition_chromosomes, composition_fitness

    def evolve_chromosome_compositions(
        self, generation_data: list[ModelRunData], allowed_gene_pool: list
    ):
        composition_chromosomes, composition_fitness = self._get_compositions_fitness(
            generation_data
        )

        self.composition_evolution_machine.set_allowed_gene_pool(allowed_gene_pool)

        new_compositions = self.composition_evolution_machine.evolution_step(
            chromosome_population=composition_chromosomes,
            population_fitness=composition_fitness,
        )
        return new_compositions

    def _get_composition_type_counts(self, generation_data: list[ModelRunData]):
        composition_type_counts = {}
        for model_run in generation_data:
            if model_run.parameter_run_number != 0:
                continue
            for agent_data in model_run.model_data.agent_data:
                if agent_data["chromosome_type"] not in composition_type_counts:
                    composition_type_counts[agent_data["chromosome_type"]] = 0
                composition_type_counts[agent_data["chromosome_type"]] += 1
        return composition_type_counts

    def _get_species_fitness_scalars(
        self, chromosome_type_counts, composition_type_counts
    ):
        species_fitness_scalars = {}
        # total chromosome count / total composition slots count, to scale the scalar
        size_scalar = sum(chromosome_type_counts.values()) / sum(
            composition_type_counts.values()
        )
        for chromosome_type in chromosome_type_counts:
            # NOTE This is a very simple formula. It can be improved. Though, it seems to work perfectly fine.
            species_fitness_scalars[chromosome_type] = (
                composition_type_counts[chromosome_type] * size_scalar
            ) / chromosome_type_counts[chromosome_type]
        # print(species_fitness_scalars)
        return species_fitness_scalars

    def _scale_species_fitness(
        self, chromosome_fitness_per_species, species_fitness_scalars
    ):
        # need a small offset to scale if the fitness is 0
        offset = 1

        # The fitness can be negative or positive, scaling needs to be done differently.
        # NOTE Some optimization possible here, this should only be done once per model run
        # as the fitness type doesn't change.
        positive_fitness = all(
            fitness >= 0
            for chromosome_species_fitness in chromosome_fitness_per_species.values()
            for _, fitness in chromosome_species_fitness
        )

        scalar_function = (
            self._scale_fitness_positive
            if positive_fitness
            else self._scale_fitness_negative
        )
        for (
            chromosome_type,
            chromosome_species_fitness,
        ) in chromosome_fitness_per_species.items():
            for idx, (genes, fitness) in enumerate(chromosome_species_fitness):
                chromosome_species_fitness[idx] = (
                    genes,
                    scalar_function(
                        fitness, species_fitness_scalars[chromosome_type], offset
                    ),
                )
        return chromosome_fitness_per_species

    def _scale_fitness_negative(self, fitness, scalar, offset=1):
        # Negative fitness values need to be scaled so they are closer to 0
        assert (
            fitness <= 0
        ), f"Fitness scaling with composition doesn't work if there are both positive and negative fitness values. Fitness: {fitness}, scale is set to negative values."
        return (fitness - offset) / scalar

    def _scale_fitness_positive(self, fitness, scalar, offset=1):
        # Positive fitness values need to be scaled so they are larger
        return (fitness + offset) * scalar

    def _scale_species_fitness_with_composition_representation(
        self, chromosome_fitness_per_species, generation_data: list[ModelRunData]
    ):
        chromosome_type_counts = {
            type_: len(chromosomes)
            for type_, chromosomes in chromosome_fitness_per_species.items()
        }
        composition_type_counts = self._get_composition_type_counts(generation_data)
        # print(f"chromosome type counts: {chromosome_type_counts}")
        # print(f"composition type counts: {composition_type_counts}")
        species_fitness_scalars = self._get_species_fitness_scalars(
            chromosome_type_counts, composition_type_counts
        )
        chromosome_fitness_per_species = self._scale_species_fitness(
            chromosome_fitness_per_species, species_fitness_scalars
        )
        return chromosome_fitness_per_species

    def generate_new_chromosome_combinations(
        self,
        new_chromosomes: list[Chromosome],
        generation_data: list[ModelRunData],
    ) -> list[list[Chromosome]]:
        new_chromosome_types = self.get_chromosome_types(new_chromosomes)
        chromosome_commpositions = self.evolve_chromosome_compositions(
            generation_data=generation_data,
            allowed_gene_pool=new_chromosome_types,
        )
        if self.align_compositions_with_population:
            chromosome_commpositions = (
                align_compositions_with_chromosome_population_types(
                    chromosome_commpositions, new_chromosomes
                )
            )

        # for composition in chromosome_commpositions:
        #     print(composition)
        chromosome_combinations = generate_chromosome_sets_from_compositions(
            chromosome_commpositions, new_chromosomes
        )
        return chromosome_combinations

    def extinction_event(self, chromosomes: list[Chromosome]) -> list[Chromosome]:
        purge_type = "single_random_proportional"
        chromosome_types = [chromosome.chromosome_type for chromosome in chromosomes]
        chromosome_types_set = set(chromosome_types)

        if purge_type == "single_random":
            species_to_purge = [
                chromosome_types_set[random.randint(0, len(chromosome_types_set) - 1)]
            ]
        elif purge_type == "single_random_proportional":
            species_to_purge = [
                chromosome_types[random.randint(0, len(chromosome_types) - 1)]
            ]
        elif purge_type == "all_but_one_random":
            species_to_keep = chromosome_types_set[
                random.randint(0, len(chromosome_types_set) - 1)
            ]
            species_to_purge = chromosome_types_set - {species_to_keep}
        elif purge_type == "all_but_one_random_proportional":
            species_to_keep = chromosome_types[
                random.randint(0, len(chromosome_types) - 1)
            ]
            species_to_purge = chromosome_types_set - {species_to_keep}
        elif purge_type == "all_but_most_common":
            species_to_keep = max(
                chromosome_types_set, key=lambda type_: chromosome_types.count(type_)
            )
            species_to_purge = chromosome_types_set - {species_to_keep}
        elif purge_type == "random_amount_of_species":
            species_to_purge = random.shuffle(list(chromosome_types_set))[
                : random.randint(0, len(chromosome_types_set) - 1)
            ]

        print(f"Purge type: {purge_type}")
        print(f"Purging species {species_to_purge}")
        for species in species_to_purge:
            print(f"Species {species} -> {chromosome_types.count(species)} chromosomes")
        return [
            chromosome
            for chromosome in chromosomes
            if chromosome.chromosome_type not in species_to_purge
        ]
