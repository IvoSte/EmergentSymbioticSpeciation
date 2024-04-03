from dataclasses import dataclass, replace
from .sampling import (
    generate_deterministic_chromosome_combinations,
    generate_chromosome_combinations,
)
from shared_components.logger import log
from shared_components.parameters import Parameters

# module used to load hyperparameter sets. Read from file or generate,
# return a list of parameters as dictionary where key= parameter, value= parameter value


# Other models require different Paramter objects.
# Worry about that later, easier to fix once you actually have an other model.
# I'm here now, lets do this.


class HyperParameters:
    def __init__(self, parameter_type: Parameters, config):
        self.default_parameter = parameter_type.from_config(config)

    def get_parameter_set_with_set_chromosomes(self, n_parameters, chromosomes):
        # Extend this function to take in multiple chromosomes, tied to type.
        parameter_set = [
            replace(self.default_parameter, agent_chromosomes=chromosomes)
            for _ in range(n_parameters)
        ]
        return parameter_set

    def generate_parameter_set_from_chromosomes(
        self, chromosomes, n_parameters, chromosomes_per_set, sampling_method
    ) -> list[Parameters]:
        chromosome_combinations = generate_chromosome_combinations(
            chromosomes=chromosomes,
            n_combinations=n_parameters,
            combination_size=chromosomes_per_set,
            method=sampling_method,
        )
        parameter_set = [
            replace(self.default_parameter, agent_chromosomes=chromosome_combination)
            for chromosome_combination in chromosome_combinations
        ]
        return parameter_set

    # def generate_parameter_set_from_chromosomes_old(self, n_parameters, chromosomes):
    #     # WARNING: Only works with equal amout of chromosomes per type
    #     # WARNING: Depricated function. The many warnings are caught with a single assert in a function further down.

    #     ## Generate some warnings
    #     chromosome_type_counts = {
    #         chromosome.chromosome_type: 0 for chromosome in chromosomes
    #     }
    #     for chromosome in chromosomes:
    #         chromosome_type_counts[chromosome.chromosome_type] += 1
    #     if len(set(chromosome_type_counts.values())) != 1:
    #         log.warning(
    #             "Chromosome types are not equal in amount. Generating parameter sets with combinations will not work or produce unexpected results."
    #         )
    #     if list(chromosome_type_counts.values())[0] > n_parameters:
    #         log.warning(
    #             f"There are more chromosomes per type ({list(chromosome_type_counts.values())[0]}) than parameter sets ({n_parameters}), meaning some ({list(chromosome_type_counts.values())[0] - n_parameters}) are not used."
    #         )
    #     if list(chromosome_type_counts.values())[0] < n_parameters:
    #         log.warning(
    #             "There are fewer chromosomes per type than parameter sets, meaning some get repeated in combinations."
    #         )
    #     ## End of warnings

    #     # TODO replace with generate chromosome combinations. problem: you don't have predator types here.
    #     chromosome_combinations = generate_deterministic_chromosome_combinations(
    #         chromosomes, n_parameters
    #     )

    #     parameter_set = []
    #     for chromosome_combination in chromosome_combinations:
    #         parameter_set.append(
    #             replace(
    #                 self.default_parameter, agent_chromosomes=chromosome_combination
    #             )
    #         )
    #     return parameter_set
