import copy
import numpy as np
from .chromosome import Chromosome
from shared_components.logger import log

# receive chromosomes, with a type
# count the types
# generate combinations of size 3 (number of foxes) of the types
# transform the combinations into a set of chromosomes
# return the set


def generate_chromosome_combinations(
    chromosomes, n_combinations, combination_size, method="uniform"
):
    assert (
        n_combinations > 0 and combination_size > 0
    ), f"n_combinations: {n_combinations}, combination_size: {combination_size} must be > 0"
    if method == "deterministic":
        chromosome_sets = generate_deterministic_chromosome_combinations(
            chromosomes, n_combinations
        )
    elif method == "uniform":
        chromosome_sets = get_chromosome_compositions(
            chromosomes, n_combinations, combination_size
        )
    elif method == "simple_random":
        chromosome_sets = generate_simple_random_chromosome_combinations(
            chromosomes, n_combinations, combination_size
        )
    elif method == "evolutionary":
        # This function should only be executed once, with the initialization of the hyperparameters. This code smells a bit. NOTE
        chromosome_sets = generate_simple_random_chromosome_combinations(
            chromosomes, n_combinations, combination_size
        )
    else:
        raise ValueError("Method not recognized")
    return chromosome_sets


def generate_evolutionary_chromosome_combinations():
    # get compositions, fitness and allowed genes
    # set allowed genes
    # evolve genenration with evmachine
    # use compositions to generate chromosome sets
    # return chromosome sets
    # This function is not made because I don't want to pass the evmachine to this module.
    pass


def get_chromosome_compositions(chromosomes, n_compositions, composition_size):
    type_counts = get_chromosome_type_counts(chromosomes)
    compositions = generate_type_compositions(
        type_counts, n_compositions, composition_size
    )
    compositions = align_compositions_with_chromosome_population_types(
        compositions, chromosomes
    )

    chromosome_sets = generate_chromosome_sets_from_compositions(
        compositions, chromosomes
    )
    return chromosome_sets


def get_chromosome_type_counts(chromosomes):
    types_counts = {}
    for chromosome in chromosomes:
        if chromosome.chromosome_type not in types_counts:
            types_counts[chromosome.chromosome_type] = 0
        types_counts[chromosome.chromosome_type] += 1
    return types_counts


def get_chromosome_per_type(chromosomes):
    chromosome_per_type = {}
    for chromosome in chromosomes:
        if chromosome.chromosome_type not in chromosome_per_type:
            chromosome_per_type[chromosome.chromosome_type] = []
        chromosome_per_type[chromosome.chromosome_type].append(chromosome)
    return chromosome_per_type


def generate_type_compositions(type_counts, n_compositions, composition_size):
    compositions = generate_uniform_distributed_type_compositions(
        type_counts, n_compositions, composition_size
    )
    # check_compositions_completeness(compositions, type_counts.keys())
    return compositions


def generate_uniform_distributed_type_compositions(
    type_counts, n_compositions, composition_size
):
    compositions = []
    type_probability = [type_counts[x] / sum(type_counts.values()) for x in type_counts]
    for _ in range(n_compositions):
        composition = []
        for _ in range(composition_size):
            type_ = np.random.choice(list(type_counts.keys()), p=type_probability)
            composition.append(type_)
        compositions.append(composition)
    return compositions


def generate_chromosome_sets_from_compositions(compositions, chromosomes):
    chromosome_sets = [[] for _ in range(len(compositions))]
    compositions_ = copy.deepcopy(compositions)
    # for each chromosome
    # Loop over all compositions and place it in the first composition that has a spot for it
    # until all slots are filled. This ensures each chromosome is used at least once when there is a spot.
    while any(compositions_):
        np.random.shuffle(chromosomes)
        for chromosome in chromosomes:
            for composition_idx, composition in enumerate(compositions_):
                if chromosome.chromosome_type in composition:
                    chromosome_sets[composition_idx].append(chromosome)
                    composition.remove(chromosome.chromosome_type)
                    break
    return chromosome_sets


def generate_deterministic_chromosome_combinations(
    chromosomes, n_combinations
) -> list[list[Chromosome]]:
    """_summary_
    Short explanation of this function:
    The algorithm loops over the equal sized sets of chromosomes of different types
    and creates as many unique combinations as possible until the requested number
    of combinations is made.
    it does so by pulling the ith + (type index * loops) from each set.
    this means that on the first loop, it grabs the first from all
    the second loop, the first from the first set, the second from the second and
    third from the third set
    The third loop the first from the first set, the third from the second, fifth from the third

    Args:
        n_combinations (int): number of combinations to make
        chromosomes (list[Chromosome]): chromosomes used to make the combinations

    Returns:
        list[list[Chromosome]]: List of chromosome combinations (list)
    """
    chromosome_per_type = get_chromosome_per_type(chromosomes)

    assert all(
        len(chromosomes) == len(list(chromosome_per_type.values())[0])
        for chromosomes in chromosome_per_type.values()
    ), "ERROR: function sampling.generate_chromosome_combinations does not work with an unequal amount of chromosomes per type."

    n_chromosomes_per_type = len(list(chromosome_per_type.values())[0])
    chromosome_combinations = []
    loops = 0

    while True:
        for chr_idx in range(n_chromosomes_per_type):
            chromosome_combination = []
            for type_idx, chromosome_type in enumerate(chromosome_per_type):
                chromosome_combination.append(
                    chromosome_per_type[chromosome_type][
                        (chr_idx + (loops * type_idx)) % (n_chromosomes_per_type)
                    ]
                )

            chromosome_combinations.append(chromosome_combination)
            if len(chromosome_combinations) == n_combinations:
                return chromosome_combinations
        loops += 1

    log.errors(
        f"Reached unreachable code in sampling.generate_chromosome_combinations()"
    )
    return chromosome_combinations


def generate_simple_random_chromosome_combinations(
    chromosomes, n_combinations, combination_size
):
    chromosome_sets = []
    chromosome_bag = chromosomes.copy()
    np.random.shuffle(chromosome_bag)
    for _ in range(n_combinations):
        chromosome_sets.append([])
        for _ in range(combination_size):
            chromosome = chromosome_bag.pop()
            chromosome_sets[-1].append(chromosome)
            if len(chromosome_bag) == 0:
                chromosome_bag = chromosomes.copy()
                np.random.shuffle(chromosome_bag)
    return chromosome_sets


def align_compositions_with_chromosome_population_types(compositions, chromosomes):
    # If a chromosome type occurs more in the chromosome population than in the compositions, change the compositions to create more slots for that type
    # We do this so that all chromosomes can be evaluated at least once.
    # We do this in the most fair way -- we check the deltas between how often a type occurs in the chromosome population
    # and in the compositions. A negative delta means that the type occurs more in the chromosome population than in the compositions, and would so be left out.
    # For each negative delta, transform a composition by replacing it with the most overrepresented type.
    type_deltas = get_chromosome_composition_type_deltas(compositions, chromosomes)

    # Get transformations
    transformations = []
    while min(type_deltas.values()) < 0:
        if max(type_deltas.values()) == 0:
            raise ValueError(
                """Problem aligning compositions with chromosome population types.
                There are more chromosomes than composition slots.
                The cause is probably that composition chromosomes are of variable length (chromosome resizing is on), and there are just enough compositions to fit all chromosomes.
                Consider increasing the number of compositions by increasing the parameter set size.
                """
            )
        from_type = max(type_deltas, key=type_deltas.get)
        to_type = min(type_deltas, key=type_deltas.get)
        type_deltas[from_type] -= 1
        type_deltas[to_type] += 1
        transformations.append((from_type, to_type))

    # Apply transformations
    for from_type, to_type in transformations:
        # Loop over the compositions randomly so we don't introduce a bias with changing earlier compositions
        np.random.shuffle(compositions)
        for composition in compositions:
            if from_type in composition:
                composition.remove(from_type)
                composition.append(to_type)
                break
    return compositions


def get_chromosome_composition_type_deltas(compositions, chromosomes):
    # Check the differences between type occurance in the chromosome population and the compositions.
    # If there are more chromosomes of a type than slots for that type, we need to change the compositions to make room for that type.
    chromosome_types = [chromosome.chromosome_type for chromosome in chromosomes]
    composition_types = [type_ for composition in compositions for type_ in composition]
    chromosome_type_counts = {
        type_: chromosome_types.count(type_) for type_ in set(chromosome_types)
    }
    composition_type_counts = {
        type_: composition_types.count(type_) for type_ in set(chromosome_types)
    }
    type_deltas = {
        type_: composition_type_counts[type_] - chromosome_type_counts[type_]
        for type_ in set(chromosome_types)
    }
    return type_deltas


# Some functions to check the validity of the generated combinations. Also known as tests. :) (smiley generated by github copilot)
def check_compositions_completeness(compositions, types):
    checker = {type_: False for type_ in types}
    for type_ in types:
        if checker[type_]:
            continue
        for composition in compositions:
            if type_ in composition:
                checker[type_] = True
                continue
    if not all(checker.values()):
        print("Not all types are represented in the compositions")


def check_all_chromosomes_present(chromosomes, combinations):
    for chromosome in chromosomes:
        if not chromosome_in_combinations(chromosome, combinations):
            print(f"Chromosome {chromosome} not in combinations")


def chromosome_in_combinations(chromosome, combinations):
    for combination in combinations:
        if chromosome in combination:
            return True
    return False


def check_chromosome_sets(chromosome_sets):
    for chromosome_set in chromosome_sets:
        if len(chromosome_set) != 3:
            log.warning(f"Chromosome set {chromosome_set} does not have 3 chromosomes")


def check_compositions_for_type_slot_completeness(compositions, chromosomes):
    # Check if there is at least one slots for all chromosomes in the compositions.
    # E.G. if there are two type B chromosomes, there should be at least two slots for type B in the compositions.
    type_deltas = get_chromosome_composition_type_deltas(compositions, chromosomes)
    for type_ in type_deltas:
        if type_deltas[type_] < 0:
            log.warning(
                f"Type {type_} has {-type_deltas[type_]} fewer slots than chromosomes"
            )


def compositions_generation_overview(compositions, chromosomes):
    type_deltas = get_chromosome_composition_type_deltas(compositions, chromosomes)
    print(f"Type deltas: {type_deltas}")

    composition_types = [type_ for composition in compositions for type_ in composition]
    composition_type_counts = {
        type_: composition_types.count(type_) for type_ in set(composition_types)
    }
    print(f"Composition type counts: {composition_type_counts}")

    composition_occurances = get_composition_occurances(compositions)
    print(f"Composition occurances: {composition_occurances}")

    # type_cooccurances = get_type_cooccurances(compositions, composition_types)
    # print(f"Type cooccurances: {type_cooccurances}")


def get_type_cooccurances(compositions, types):
    type_cooccurances = {}
    for type_ in types:
        type_cooccurances[type_] = {}
        for type_b in types:
            type_cooccurances[type_][type_b] = 0
    for composition in compositions:
        for type_a in composition:
            for type_b in composition:
                if type_a != type_b:
                    type_cooccurances[type_a][type_b] += 1
    return type_cooccurances


def get_composition_occurances(compositions):
    composition_occurances = {}
    for composition in compositions:
        composition = tuple(sorted(composition))
        if composition not in composition_occurances:
            composition_occurances[composition] = 0
        composition_occurances[composition] += 1

    composition_occurances = dict(
        sorted(composition_occurances.items(), key=lambda item: item[1], reverse=True)
    )
    return composition_occurances


### depricated or unmade functions that I don't want to delete yet ###

# class Sampler:
#     def __init__(self, method="uniform"):
#         self.method = method

#     def generate_chromosome_combinations(self):
#         pass
# Here rests a good idea that was not implemented to save time

# def generate_stratified_chromosome_combinations(chromosomes, n_combinations, combination_size):


def generate_chromosome_sets_from_compositions_old(compositions, chromosomes):
    # Alternative implementation, not used
    chromosome_sets = []
    chromosome_bag = chromosomes.copy()
    for composition in compositions:
        chromosome_sets.append([])
        for type_ in composition:
            # pick a random chromosome of the correct type and put it in the last set
            try:
                chromosome = np.random.choice(
                    [
                        chromosome
                        for chromosome in chromosome_bag
                        if chromosome.chromosome_type == type_
                    ]
                )
                # remove the chromosome from the bag
                chromosome_bag.remove(chromosome)
            except ValueError:
                # If there are no chromosomes left of that type in the current bag, get one from the big pool.
                # NOTE ATTENTION: This is a design choice, it might cause an imbalance in chromosome uses,
                # possibly a slight overrepresentation of novel types, but that seems fine, or even desired.
                chromosome = np.random.choice(
                    [
                        chromosome
                        for chromosome in chromosomes
                        if chromosome.chromosome_type == type_
                    ]
                )
            # add the chromosome to the last set
            chromosome_sets[-1].append(chromosome)
            # if the bag is empty, replenish the bag
            if len(chromosome_bag) == 0:
                log.debug("used all chromosomes, replenishing bag")
                chromosome_bag = chromosomes.copy()
    return chromosome_sets


if __name__ == "__main__":
    type_counts = {"A": 3, "B": 5, "C": 1}
    n_compositions = 1000
    composition_size = 3
    result = generate_type_compositions(type_counts, n_compositions, composition_size)
    counts = {"A": 0, "B": 0, "C": 0}
    for comp in result:
        for type_ in comp:
            counts[type_] += 1
    # print(result)
    print(counts)
