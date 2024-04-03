import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout


class SpeciesGraph:
    def __init__(self, species):
        self.species = species

    def _species_to_input_format(self):
        self.species_tuples = []
        species = self.species.copy()
        for species in self.species.values():
            for generation in species.generations:
                self.species_tuples.append(
                    (species.permanent_name, species.parent_species_name, generation)
                )
                species.parent_species_name = species.permanent_name
        return self.species_tuples

    def plot(self):
        species_data = self._species_to_input_format()
        self.plot_phylogenetic_tree_simple(species_data)

    def plot_phylogenetic_tree(self, species_data):
        # Create a directed graph
        G = nx.DiGraph()

        for species in species_data:
            # species_data format: [(species_name, origin, generation), ...]
            species_name, origin, generation = species

            # Add node for the species
            G.add_node(species_name, generation=generation)

            # Add an edge from the origin to the species
            if origin is not None:
                G.add_edge(origin, species_name)

        # Use the generation attribute to get the x-position for each node
        pos = graphviz_layout(G, prog="dot")

        # Draw the graph
        plt.figure(figsize=(10, 10))
        nx.draw(
            G,
            pos,
            with_labels=True,
            arrows=False,
            node_size=1000,
            node_color="lightblue",
        )

        plt.show()

    def plot_phylogenetic_tree_simple(self, species_data):
        # Create a directed graph
        G = nx.DiGraph()

        for species in species_data:
            # species_data format: [(species_name, origin, generation), ...]
            species_name, origin, generation = species

            # Add node for the species
            G.add_node(species_name, generation=generation)

            # Add an edge from the origin to the species
            if origin is not None:
                G.add_edge(origin, species_name)

        # Use the spring_layout function to get the position for each node
        pos = nx.spring_layout(G)

        # Draw the graph
        plt.figure(figsize=(10, 10))
        nx.draw(
            G,
            pos,
            with_labels=True,
            arrows=False,
            node_size=1000,
            node_color="lightblue",
        )

        plt.show()


def plot_example():
    import matplotlib.pyplot as plt
    import numpy as np

    # Suppose data is a dictionary where keys are species names
    # (converted to integers) and values are lists of generations in which species appeared
    data = {1: [1, 2, 3], 2: [2, 3, 4], 3: [3, 4, 5], 4: [4, 5, 6]}

    # Also suppose you have another dictionary that specifies parent-child relationships
    parent_child = {1: [], 2: [1], 3: [2], 4: [3]}

    fig, ax = plt.subplots()

    # Loop through each species
    for species, generations in data.items():
        # Create a scatter plot for the species
        ax.scatter(generations, np.full_like(generations, fill_value=species))

        # Loop through each generation
        for i in range(len(generations) - 1):
            # If the species has a parent
            if parent_child[species]:
                # Draw a line to the parent's position in the previous generation
                ax.plot(
                    [generations[i], generations[i + 1]],
                    [species, parent_child[species][0]],
                    color="black",
                )

    # Set the yticks to the species names
    ax.set_yticks(list(data.keys()))
    ax.set_yticklabels(["Species {}".format(i) for i in data.keys()])

    # Set the xticks to the generations
    ax.set_xticks(range(1, max([max(v) for v in data.values()]) + 1))
    ax.set_xticklabels(range(1, max([max(v) for v in data.values()]) + 1))

    # Label the axes
    ax.set_xlabel("Generation")
    ax.set_ylabel("Species")

    plt.show()


if __name__ == "__main__":
    plot_example()
