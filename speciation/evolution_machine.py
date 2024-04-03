from dataclasses import dataclass
from math import ceil
import random
import numpy as np
from enum import Enum

# Note:
# This code is not perfect, there are some things that are slightly off but work for now. When reusing this code, please keep the following in mind (or improve it)
# - The number of parents (n_reproduce) is set to an even value for convenience. This will run into errors for reproduce_fraction  = 1,
#   as there will be an index out of range error in the produce_offspring() function.


# Legacy: Earlier, it was not possible to have reproduce_fraction and cull_fraction summed exceed 1, but better code behaviour fixed this.
# An artifact of that earlier code is that at one point n_cull needed to be even. Here the error would also occur that n_cull could exceed
# the population size. This is fixed by returning a sliced list from produce_offspring(). It is possible a child is created that is then discarded. (sad)

# Note 2: I believe the comments above are no longer valid


@dataclass
class Genome:
    genes: np.ndarray
    fitness: float

    def __eq__(self, other):
        return np.all(self.genes == other.genes) and self.fitness == other.fitness

    def distance(self, other):
        return sum(
            [abs(self.genes[i] - other.genes[i]) for i in range(len(self.genes))]
        )
        # return np.linalg.norm(self.genes - other.genes)


@dataclass
class GenomeNominal(Genome):
    genes: list
    fitness: float

    def __eq__(self, other):
        return self.genes == other.genes and self.fitness == other.fitness

    def distance(self, other):
        return sum(
            [1 for i in range(len(self.genes)) if self.genes[i] != other.genes[i]]
        )


@dataclass
class GenomeBinary(Genome):
    genes: list[bool]
    fitness: float

    def __eq__(self, other):
        return self.genes == other.genes and self.fitness == other.fitness

    def distance(self, other):
        return sum(
            [1 for i in range(len(self.genes)) if self.genes[i] != other.genes[i]]
        )


class SelectionType(Enum):
    RANDOM = "random"  # Random should be combined with reproduce_fraction < 1.0, to make it truncation selection.
    SEQUENTIAL = "sequential"  # Sequential means that the first parents will be selected sequentially from the ranked population, meaning the fittest individual are selected first, but without repetition until all individuals have been selected.
    FITNESS_PROPORTIONAL = "fitness_proportional"


class MateSelectionType(Enum):
    RANDOM = "random"
    FITNESS_PROPORTIONAL = "fitness_proportional"
    NEAREST_NEIGHBOUR = "nearest_neighbour"


class GeneDataType(Enum):
    CONTINUOUS = "continuous"
    NOMINAL = "nominal"
    BINARY = "binary"


class CrossoverType(Enum):
    UNIFORM = "uniform"
    UNIFORM_VARIABLE_LENGTH = "uniform_variable_length"
    SINGLE_POINT = "single_point"


class EvolutionMachine:
    """An evolutionary machine is a class that can be used to evolve a population of chromosomes.
    Usage:  Initialize the machine with the parameters you want to use. Then call the evolution_step() function with the current population and fitness values.
            The function will return a new set of chromosomes that can be used as the next population.
    """

    def __init__(
        self,
        reproduce_fraction: float = 0.5,
        cull_fraction: float = 0.5,
        mutation_probability: float = 0.1,
        mutation_range: float = 0.1,
        expected_population_size: int = None,
        selection_type: SelectionType = SelectionType.RANDOM,
        mate_selection: MateSelectionType = MateSelectionType.RANDOM,
        mate_selection_sample_size: int = 0,
        allow_self_mating: bool = False,
        gene_data_type: GeneDataType = GeneDataType.CONTINUOUS,
        knockout_mutation: bool = False,
        knockout_mutation_probability: float = 0.1,
        chromosome_resizing: bool = False,
        chromosome_resizing_probability: float = 0.1,
        crossover_type: CrossoverType = CrossoverType.UNIFORM,
    ):
        """
        Args:
            reproduce_fraction (float): The fraction of the population that will be considered for reproduction. Must be between 0 and 1. Truncation Selection parameter.
            cull_fraction (float): The fraction of the population that will be culled. Must be between 0 and 1. The inverse of the Elitism parameter.
            mutation_probability (float): The probability that a gene will mutate. Must be between 0 and 1.
            mutation_range (float): The range of the mutation. New gene value = old gene value + random(-mutation_range, mutation_range)
            expected_population_size (int): Control the output population size. If None, returns a population of the same size as the input population. Cull and reproduce values are scaled with this number, to be consistent with chromosome bleeding.
            selection_type (SelectionType, optional): The selection type. Defaults to RANDOM. Can alos be SEQUENTIAL or FITNESS_PROPORTIONAL.
            mate_selection (MateSelectionType, optional): The mate selection type. Defaults to RANDOM. Can also be FITNESS_PROPORTIONAL or NEAREST_NEIGHBOUR.
            mate_selection_sample_size (int, optional): The sample size for mate selection. Defaults to 0. 0 means that all individuals are considered.
            allow_self_mating (bool, optional): Whether to allow self mating. Defaults to False.
        """
        assert (
            reproduce_fraction <= 1.0 and reproduce_fraction >= 0.0
        ), "Evolutionary machines reproduce fraction must be between 0 and 1."
        assert (
            cull_fraction <= 1.0 and cull_fraction >= 0.0
        ), "Evolutionary machines cull fraction must be between 0 and 1."
        assert (mutation_probability <= 1.0) and (
            mutation_probability >= 0.0
        ), "Mutation probability must be between 0 and 1."

        self.reproduce_fraction = reproduce_fraction
        self.cull_fraction = cull_fraction

        self.mutation_probability = mutation_probability
        self.mutation_range = mutation_range

        self.expected_population_size = expected_population_size

        self.selection_type = selection_type
        self.mate_selection = mate_selection
        self.mate_selection_sample_size = mate_selection_sample_size
        self.allow_self_mating = allow_self_mating

        self.knockout_mutation = knockout_mutation
        self.knockout_mutation_probability = knockout_mutation_probability

        self.chromosome_resizing = chromosome_resizing
        self.chromosome_resizing_probability = chromosome_resizing_probability

        self.crossover_type = crossover_type

        self.gene_data_type = gene_data_type
        if self.knockout_mutation:
            assert (
                self.knockout_mutation_probability <= 1.0
                and self.knockout_mutation_probability >= 0.0
            ), "Knockout mutation probability must be between 0 and 1."
            assert (
                self.gene_data_type == GeneDataType.CONTINUOUS
            ), "Knockout mutation is only available for continuous genes."

        self.genome_type = self._get_genome_type()
        self.allowed_gene_pool = None

    def evolution_step(self, chromosome_population, population_fitness):
        # Transform population into a list of genomes, make sure they are sorted on fitness
        genome_population = [
            self.genome_type(chromosome, fitness)
            for chromosome, fitness in zip(chromosome_population, population_fitness)
        ]
        genome_population = sorted(
            genome_population, key=lambda x: x.fitness, reverse=True
        )

        # get the number of individuals to reproduce and cull
        # n_reproduce = ceil(self.reproduce_fraction * len(genome_population))
        # n_cull = ceil(self.cull_fraction * len(genome_population))
        # n_offspring = n_cull

        true_population_size = len(genome_population)
        expected_population_size = (
            self.expected_population_size
            if self.expected_population_size
            else len(genome_population)
        )
        missing_population = expected_population_size - true_population_size
        n_reproduce = min(
            ceil(self.reproduce_fraction * expected_population_size),
            true_population_size,
        )
        n_cull = max(
            0, ceil(self.cull_fraction * expected_population_size) - missing_population
        )
        n_offspring = n_cull + missing_population

        assert (
            n_reproduce > 0 and n_offspring > 0
        ), f"Problem in evolution_step(), n_reproduce or n_offspring is 0. {n_reproduce = } {n_offspring = }"

        # Cull bottom percentage
        ranked_population = [genome.genes for genome in genome_population]
        culled_population = self._cull_population(ranked_population, n_cull)

        # Produce offspring
        offspring = self._produce_offspring(
            genome_population=genome_population,
            n_reproduce=n_reproduce,
            n_offspring=n_offspring,
        )
        new_population = culled_population + offspring
        # Check that the new population is the same size as the old population
        assert (
            len(new_population) == len(chromosome_population)
            or len(new_population) == expected_population_size
        ), f"Problem in evolution_step(), new population not the same size ({len(new_population)}) as input population ({len(ranked_population)} or expected population size ({expected_population_size}))\n {self.reproduce_fraction = } {self.cull_fraction = }"
        return new_population

    def _cull_population(self, population, n_cull):
        return population[0 : len(population) - n_cull]

    def _produce_offspring(
        self, genome_population: list[Genome], n_reproduce: int, n_offspring: int
    ):
        offspring = []

        n_parent_pairs = ceil(n_offspring / 2)

        # select parents for reproduction
        parents_zipped = self._select_reproduction_pairs(
            genome_population, n_reproduce, n_parent_pairs
        )

        # create offspring from the parents
        for parent_1, parent_2 in parents_zipped:
            offspring_1, offspring_2 = self._crossover(parent_1.genes, parent_2.genes)
            offspring.append(offspring_1)
            offspring.append(offspring_2)

        # mutate offspring
        for child in offspring:
            child = self._mutate_chromosome(child)
            if self.chromosome_resizing:
                child = self._resize_chromosome(child)
        return offspring[:n_offspring]

    def _crossover(self, parent_1, parent_2) -> tuple[list, list]:
        # Produce two offsprings based on two parents, both getting the other gene of the parents.
        if self.crossover_type == CrossoverType.UNIFORM:
            return self._uniform_crossover(parent_1, parent_2)
        elif self.crossover_type == CrossoverType.UNIFORM_VARIABLE_LENGTH:
            return self._uniform_crossover_with_variable_length_chromosomes(
                parent_1, parent_2
            )
        elif self.crossover_type == CrossoverType.SINGLE_POINT:
            return self._single_point_crossover(parent_1, parent_2)

    def _uniform_crossover(self, parent_1, parent_2) -> tuple[list, list]:
        assert len(parent_1) == len(
            parent_2
        ), f"Length of parent chromosomes unequal at crossover step."
        offspring_1 = []
        offspring_2 = []
        for idx in range(len(parent_1)):
            if random.random() < 0.5:
                offspring_1.insert(idx, parent_1[idx])
                offspring_2.insert(idx, parent_2[idx])
            else:
                offspring_1.insert(idx, parent_2[idx])
                offspring_2.insert(idx, parent_1[idx])

        return (offspring_1, offspring_2)

    def _uniform_crossover_with_variable_length_chromosomes(
        self, parent_1, parent_2
    ) -> tuple[list, list]:
        offspring_1 = []
        offspring_2 = []
        for idx in range(max(len(parent_1), len(parent_2))):
            if random.random() < 0.5:
                if idx < len(parent_1):
                    offspring_1.append(parent_1[idx])
                if idx < len(parent_2):
                    offspring_2.append(parent_2[idx])
            else:
                if idx < len(parent_2):
                    offspring_1.append(parent_2[idx])
                if idx < len(parent_1):
                    offspring_2.append(parent_1[idx])
        return (offspring_1, offspring_2)

    def _single_point_crossover(self, parent_1, parent_2) -> tuple[list, list]:
        offspring_1 = []
        offspring_2 = []
        crossover_point = random.randint(0, min(len(parent_1), len(parent_2)))
        offspring_1 = parent_1[:crossover_point] + parent_2[crossover_point:]
        offspring_2 = parent_2[:crossover_point] + parent_1[crossover_point:]
        return (offspring_1, offspring_2)

    def _mutate_chromosome(self, chromosome):
        if self.gene_data_type == GeneDataType.CONTINUOUS:
            chromosome = self._mutate_chromosome_continuous_genes(chromosome)
        elif self.gene_data_type == GeneDataType.NOMINAL:
            chromosome = self._mutate_chromosome_nominal_genes(chromosome)
        elif self.gene_data_type == GeneDataType.BINARY:
            chromosome = self._mutate_chromosome_binary_genes(chromosome)
        return chromosome

    def _mutate_chromosome_continuous_genes(self, chromosome):
        # The mutation probability counts for each gene, so we can mutate multiple genes on the same chromosome the same time if we wish.
        for idx in range(len(chromosome)):
            if self.knockout_mutation:
                if not self._knockout_mutation(chromosome, idx):
                    # if gene is knocked-out, we do not allow it to mutate
                    continue

            if random.random() < self.mutation_probability:
                mutation_step = random.uniform(
                    -1 * self.mutation_range, self.mutation_range
                )
                chromosome[idx] += mutation_step
        return chromosome

    def _knockout_mutation(self, chromosome, idx):
        allow_gene_to_mutate = True
        if random.random() < self.knockout_mutation_probability:
            if chromosome[idx] != 0 and sum(chromosome) != chromosome[idx]:
                # Don't knock out a gene that is already knocked out, or if it is the last non-knocked out gene.
                # Knocking out means setting the gene to 0
                chromosome[idx] = 0
                allow_gene_to_mutate = False
            elif chromosome[idx] == 0:
                # Un-knocking out means selecting a random other gene value.
                chromosome[idx] = chromosome[random.randint(0, len(chromosome) - 1)]
        elif chromosome[idx] == 0:
            allow_gene_to_mutate = False
        return allow_gene_to_mutate

    def _mutate_chromosome_nominal_genes(self, chromosome):
        if self.allowed_gene_pool is None:
            raise ValueError(
                "Allowed gene pool is not set. Please use set_allowed_gene_pool() to set the allowed gene pool."
            )
        for idx in range(len(chromosome)):
            if random.random() < self.mutation_probability:
                chromosome[idx] = random.choice(self.allowed_gene_pool)
            # If there is an updating gene pool, we need to swap out genes that are not in the allowed gene pool
            elif chromosome[idx] not in self.allowed_gene_pool:
                chromosome[idx] = random.choice(self.allowed_gene_pool)
        return chromosome

    def _mutate_chromosome_binary_genes(self, chromosome):
        for idx in range(len(chromosome)):
            if random.random() < self.mutation_probability:
                chromosome[idx] = int(not chromosome[idx])
        return chromosome

    def _resize_chromosome(self, chromosome):
        if random.random() < self.chromosome_resizing_probability:
            if random.random() < 0.5:
                insert_position = random.randint(0, len(chromosome))
                if (
                    self.gene_data_type == GeneDataType.NOMINAL
                ):  # TODO this needs to be a toggle somewhere, or a parameter
                    copy_gene_value = random.choice(self.allowed_gene_pool)
                else:
                    copy_gene_value = chromosome[random.randint(0, len(chromosome) - 1)]
                chromosome.insert(insert_position, copy_gene_value)
            else:
                if len(chromosome) > 1:
                    remove_position = random.randint(0, len(chromosome) - 1)
                    chromosome.pop(remove_position)
        return chromosome

    def set_allowed_gene_pool(self, allowed_gene_pool: list):
        self.allowed_gene_pool = allowed_gene_pool

    def _select_reproduction_pairs(
        self, genome_population: list[Genome], n_reproduce: int, n_parent_pairs: int
    ):
        # limit the sample of parents if we are using a set portion of the population, aka Truncation Selection
        # Since the genome population is sorted on fitness, we can take the first n_reproduce individuals
        parent_population = genome_population[0:n_reproduce]

        first_parents = self._select_first_parents(parent_population, n_parent_pairs)
        second_parents = self._select_second_parents(parent_population, first_parents)
        return zip(first_parents, second_parents)

    def _calculate_fitness_probabilities(self, parents_fitness):
        # calculate fitness probabilities
        if min(parents_fitness) < 0:
            make_positive_term = abs(min(parents_fitness))
            parents_fitness = [
                fitness + make_positive_term for fitness in parents_fitness
            ]
        if sum(parents_fitness) == 0:
            fitness_probability = [1.0 / len(parents_fitness)] * len(parents_fitness)
        else:
            fitness_probability = [
                fitness / sum(parents_fitness) for fitness in parents_fitness
            ]
        return fitness_probability

    def _select_first_parents(
        self, parent_population: list[Genome], n_first_parents: int
    ) -> list[Genome]:
        first_parents = []
        fitness_probability = self._calculate_fitness_probabilities(
            [parent.fitness for parent in parent_population]
        )

        # select parents randomly weighted by fitness
        if self.selection_type == SelectionType.FITNESS_PROPORTIONAL:
            first_parents_idx = np.random.choice(
                len(parent_population),
                size=n_first_parents,
                p=fitness_probability,
                replace=True,
            )
            first_parents = [parent_population[idx] for idx in first_parents_idx]

        # select parents sequentially until we have enough n_first_parents
        elif self.selection_type == SelectionType.SEQUENTIAL:
            for idx in range(n_first_parents):
                first_parents.append(parent_population[idx % len(parent_population)])

        # select parents randomly without weighting by fitness
        elif self.selection_type == SelectionType.RANDOM:
            first_parents_idx = np.random.choice(
                len(parent_population), size=n_first_parents, replace=True
            )
            first_parents = [parent_population[idx] for idx in first_parents_idx]

        else:
            raise ValueError(f"Invalid selection method {self.selection_type}")
        return first_parents

    def _select_second_parents(
        self, parent_population: list[Genome], first_parents: list[Genome]
    ) -> list[Genome]:
        second_parents = []
        for parent_1 in first_parents:
            second_parents.append(self._select_mate(parent_1, parent_population.copy()))
        return second_parents

    def _select_mate(self, parent_1: Genome, parent_population: list[Genome]) -> Genome:
        eligible_parents = self._select_eligible_parents(parent_1, parent_population)
        fitness_probability = self._calculate_fitness_probabilities(
            [parent.fitness for parent in eligible_parents]
        )

        # select a mate from the eligible parents based on the mate selection method
        if self.mate_selection == MateSelectionType.RANDOM:
            return np.random.choice(eligible_parents)
        if self.mate_selection == MateSelectionType.FITNESS_PROPORTIONAL:
            return np.random.choice(eligible_parents, p=fitness_probability)
        if self.mate_selection == MateSelectionType.NEAREST_NEIGHBOUR:
            return self._select_nearest_neighbour(
                parent_1, [parent for parent in eligible_parents]
            )
        else:
            raise ValueError(f"Invalid mate selection method: {self.mate_selection}")

    def _select_eligible_parents(
        self, parent_1: Genome, parent_population: list[Genome]
    ) -> list[Genome]:
        # filter parent 1 from the eligible parents
        if not self.allow_self_mating:
            parent_population.remove(parent_1)

        # take a sample of the eligible parents
        if (
            self.mate_selection_sample_size != 0
            and self.mate_selection_sample_size < len(parent_population)
        ):
            eligible_parents = random.sample(
                parent_population, k=self.mate_selection_sample_size
            )
        else:
            eligible_parents = parent_population
        assert (
            len(eligible_parents) > 0
        ), "No eligible parents. -- Most likely parent population is singular and self mating is not allowed."
        return eligible_parents

    def _select_nearest_neighbour(
        self, parent_1: Genome, eligible_parents: list[Genome]
    ) -> Genome:
        if parent_1 in eligible_parents:
            eligible_parents.remove(parent_1)
        assert (
            len(eligible_parents) > 0
        ), "Problem in nearest neighbour selection: No eligible parents to select from after removing self. -- Suggest increasing reproduce fraction or mate selection sample size."
        return min(
            eligible_parents,
            key=lambda parent_2: parent_1.distance(
                parent_2
            ),  # self._get_distance(parent_1.genes, parent_2.genes),
        )

    def _get_distance(self, vec_1, vec_2):
        # calculate the distance between two vecs
        return sum([abs(vec_1[i] - vec_2[i]) for i in range(len(vec_1))])

    def locality_sensitive_hashing(self, vectors, k):
        # TODO Implement locality sensitive hashing
        pass

    def _get_genome_type(self):
        if self.gene_data_type == GeneDataType.CONTINUOUS:
            return Genome
        if self.gene_data_type == GeneDataType.NOMINAL:
            return GenomeNominal
        if self.gene_data_type == GeneDataType.BINARY:
            return GenomeBinary
        else:
            raise ValueError(f"Invalid gene data type: {self.gene_data_type}")
