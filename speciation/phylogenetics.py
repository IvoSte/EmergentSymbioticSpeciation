from speciation.chromosome import Chromosome
from copy import deepcopy


class PhylogeneticTree:
    def __init__(self, species_detector):
        self.species_detector = species_detector
        self.chromosome_generations = []

    def add_chromosme_generation(self, chromosome_generation: list[Chromosome]):
        self.chromosome_generations.append(deepcopy(chromosome_generation))

    def species_detection_consistency_test(self, generation):
        if generation == 0:
            print("No parent population for generation 0.")
            return

        gene_pop_a = [
            chromosome.genes
            for chromosome in self.chromosome_generations[generation]
            + self.chromosome_generations[generation - 1]
        ]
        gene_pop_b = [
            chromosome.genes
            for chromosome in self.chromosome_generations[generation]
            + self.chromosome_generations[generation + 1]
        ]
        detected_species_a = self.species_detector.get_quasispecies(gene_pop_a)
        detected_species_b = self.species_detector.get_quasispecies(gene_pop_b)
        chromosomes_in_a_generation = len(self.chromosome_generations[generation])

        print("Comparing g + g+1 with g + g-1")
        self.compare_predictions(
            detected_species_a[:chromosomes_in_a_generation],
            detected_species_b[:chromosomes_in_a_generation],
        )
        # print("Comparing g + g-1 with g")
        # self.compare_predictions(
        #     detected_species_a[chromosomes_in_a_generation:],
        #     [
        #         chromosome.chromosome_type
        #         for chromosome in self.chromosome_generations[generation]
        #     ],
        # )

    def compare_predictions(self, detected_species_a, detected_species_b):
        count_same = 0
        count_diff = 0
        translation_counts = {}
        for pred_a, pred_b in zip(
            detected_species_a,
            detected_species_b,
        ):
            if pred_a == pred_b:
                count_same += 1
            else:
                count_diff += 1
                if pred_a not in translation_counts:
                    translation_counts[pred_a] = {}
                if pred_b not in translation_counts[pred_a]:
                    translation_counts[pred_a][pred_b] = 0
                translation_counts[pred_a][pred_b] += 1

        print(f"Same: {count_same}, diff: {count_diff}")
        print(f"Same: {count_same / (count_same + count_diff)}")
        print(f"Species delta ")
        print(f"Translations: {translation_counts}")

    def get_generation_species_prototypes(self, generation):
        genes_per_type = {}
        for chromosome in self.chromosome_generations[generation]:
            if chromosome.chromosome_type not in genes_per_type:
                genes_per_type[chromosome.chromosome_type] = []
            genes_per_type[chromosome.chromosome_type].append(chromosome.genes)

        prototypes = {}
        for chromosome_type, type_genes_set in genes_per_type.items():
            prototypes[chromosome_type] = self.species_detector.get_prototype(
                type_genes_set
            )
        return prototypes

    def calculate_distances_to_prototypes(self, generation):
        prototypes = self.get_generation_species_prototypes(generation)
        distances = {}
        for chromosome in self.chromosome_generations[generation]:
            distance = self.species_detector.genomic_distance(
                chromosome.genes, prototypes[chromosome.chromosome_type]
            )
            if chromosome.chromosome_type not in distances:
                distances[chromosome.chromosome_type] = []
            distances[chromosome.chromosome_type].append(distance)
        print("Chromosome with type distance to its type prototype:")
        for type_ in distances:
            print(f"Type {type_}: {sum(distances[type_]) / len(distances[type_])}")

    def calculate_distances_between_prototypes(self, generation):
        prototypes = self.get_generation_species_prototypes(generation)
        distances = {type_: {} for type_ in prototypes}
        for type_a in prototypes:
            for type_b in prototypes:
                if type_a == type_b:
                    continue
                distance = self.species_detector.genomic_distance(
                    prototypes[type_a], prototypes[type_b]
                )
                distances[type_a][type_b] = distance

        print("Distances between type prototypes:")
        for type_ in distances:
            if distances[type_] == {}:
                return
            print(
                f"Type {type_}: min {min(distances[type_].values()):.3f}, max {max(distances[type_].values()):.3f}, avg {sum(distances[type_].values()) / len(distances[type_].values()):.3f}"
            )

    def current_species_distribution(self, generation):
        type_counts = {}
        for chromosome in self.chromosome_generations[generation]:
            if chromosome.chromosome_type not in type_counts:
                type_counts[chromosome.chromosome_type] = 0
            type_counts[chromosome.chromosome_type] += 1

        print("Species distribution:")
        for type_ in type_counts:
            print(f"Type {type_}: {type_counts[type_]}")
