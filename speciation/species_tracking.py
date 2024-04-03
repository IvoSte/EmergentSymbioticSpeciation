# If we want to detect species with knowlegde of previous generations, we can track species (prototypes) over generations.
from dataclasses import dataclass


@dataclass
class Prototype:
    prototype: list
    temp_name: str
    species_name: str
    cluster_size: int
    closest_distance: float = None
    parent_species_name: str = None


@dataclass
class Species:
    permanent_name: str
    prototypes: list
    generations: list
    parent_species_name: str = None

    def add_prototype(self, prototype, generation):
        self.prototypes.append(prototype)
        self.generations.append(generation)


class SpeciesTracker:
    def __init__(self, search_depth=1, distance_threshold=0.5):
        self.species = {}
        self.prototypes_per_generation = []
        self.generation_search_depth = search_depth
        self.prototype_distance_threshold = distance_threshold
        self.name_generator = self._generate_name()
        self.counter = {
            "threshold": 0,
            "split": 0,
        }

    def reset(self):
        self.species = {}
        self.prototypes_per_generation = []
        self.name_generator = self._generate_name()
        self.counter = {
            "threshold": 0,
            "split": 0,
        }

    def get_prototype_species(self, type_prototypes, type_cluster_sizes):
        self._add_prototype_generation(type_prototypes, type_cluster_sizes)
        self._assign_prototypes_to_species()
        self._handle_assignment_conflicts()
        self._add_prototypes_to_species(self.prototypes_per_generation[-1])
        translation_key = {
            prototype.temp_name: prototype.species_name
            for prototype in self.prototypes_per_generation[-1]
        }
        # print(f"Translation key: {translation_key}")

        # if len(self.prototypes_per_generation) == 49:
        #     self.print_species()

        # print(self.counter)
        return translation_key

    def _add_prototype_generation(self, type_prototypes, type_cluster_sizes):
        self.prototypes_per_generation.append([])
        for type_, prototype in type_prototypes.items():
            self.prototypes_per_generation[-1].append(
                Prototype(
                    prototype=prototype,
                    temp_name=type_,
                    species_name=None,
                    cluster_size=type_cluster_sizes[type_],
                )
            )

    def _assign_prototypes_to_species(self):
        for prototype in self.prototypes_per_generation[-1]:
            closest_prototype, closest_distance = self._find_closest_prototype(
                prototype
            )
            if closest_prototype is None:
                prototype.species_name = next(self.name_generator)
                self.counter["threshold"] += 1
            else:
                prototype.species_name = closest_prototype.species_name
                prototype.closest_distance = closest_distance

    def _find_closest_prototype(self, prototype: Prototype):
        closest_prototype = None
        closest_distance = None

        search_depth = (
            -len(self.prototypes_per_generation)
            if self.generation_search_depth == 0
            else -self.generation_search_depth
        )

        for previous_generation_prototypes in self.prototypes_per_generation[
            search_depth - 1 : -1
        ]:
            for previous_prototype in previous_generation_prototypes:
                distance = self._prototype_distance(
                    prototype.prototype, previous_prototype.prototype
                )
                if (
                    closest_distance is None
                    or distance < closest_distance
                    and distance < self.prototype_distance_threshold
                ):
                    closest_distance = distance
                    closest_prototype = previous_prototype
        return closest_prototype, closest_distance

    def _handle_assignment_conflicts(self):
        self._handle_species_split()

    def _handle_species_split(self):
        # If more than one species are assigned to the same species, the species is split.
        assigned_species = [
            prototype.species_name for prototype in self.prototypes_per_generation[-1]
        ]
        split_species = [
            species_name
            for species_name in assigned_species
            if assigned_species.count(species_name) > 1
        ]

        for species_name in set(
            split_species
        ):  # NOTE added this set casting because I think its correct, but if problems emerge check here. Remove this comment if they dont.
            self._split_species(species_name)

    def _split_species(self, species_name):
        # List of prototypes of the last generation that belong to the species
        prototype_species_subset = [
            prototype
            for prototype in self.prototypes_per_generation[-1]
            if prototype.species_name == species_name
        ]
        # Find the prototype with the largest cluster size, with distance as a tie-breaker
        best_match_index, _ = max(
            enumerate(prototype_species_subset),
            key=lambda prototype: (
                prototype[1].cluster_size,
                -prototype[1].closest_distance,
            ),
        )
        prototype_species_subset.pop(best_match_index)
        # Rename the remaining prototypes
        for prototype in prototype_species_subset:
            prototype.species_name = next(self.name_generator)
            self.counter["split"] += 1

    def _add_prototypes_to_species(self, prototypes: list[Prototype]):
        for prototype in prototypes:
            self._add_prototype_to_species(prototype)

    def _add_prototype_to_species(self, prototype: Prototype):
        if prototype.species_name not in self.species:
            species = Species(
                permanent_name=prototype.species_name,
                prototypes=[],
                generations=[],
                parent_species_name=prototype.parent_species_name,
            )
            self.species[prototype.species_name] = species

        self.species[prototype.species_name].add_prototype(
            prototype=prototype.prototype,
            generation=len(self.prototypes_per_generation) - 1,
        )

    def _generate_name(self):
        i = 0
        while True:
            yield str(i)
            i += 1

    def _prototype_distance(self, prototype_a, prototype_b):
        return sum(
            [abs(gene_a - gene_b) for gene_a, gene_b in zip(prototype_a, prototype_b)]
        )

    def print_species(self):
        for s in self.species.values():
            print(
                f"""
                  Species name: {s.permanent_name}
                  Prototypes: {len(s.prototypes)}
                  Generations: {len(s.generations)} -> {s.generations}
                  Parent Species: {s.parent_species_name}"""
            )

    def ghost_main(self):
        pass
        # Get the species from the current + previous generation population
        # For each species, calculate the centroid
        # For each centroid, calculate the distance to the centroids of the previous generation
        # If the distance is below a threshold, assign the species to the previous generation species
        # If ... many other cases, do other things
        # For each species, set the type of each individual to the species name of the centroid.

        # Other things that need to be done:
        # Generate names
        # Track / store species
        # For all edge cases, make sure the species are assigned correctly
