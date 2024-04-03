import os
import psutil
from models.model_factory import ModelFactory, ModelType
from speciation.species_graph import SpeciesGraph


class ModelPlotGenerator:
    def __init__(self, config):
        self.model_type = ModelType(config["MODEL"])
        self.model_factory = ModelFactory(self.model_type)
        self.config = config

    def visualize_results(self, data, separate_images=False):
        if self.model_type == ModelType.PREDATOR_PREY:
            return self._visualize_predator_prey(data, separate_images)
        elif self.model_type == ModelType.TOXIN:
            return self._visualize_toxin_model(data, separate_images)
        elif self.model_type == ModelType.FUNCTION_OPTIMIZATION:
            return self._visualize_function_optimization(data, separate_images)

    def visualize_species_graph(self, tracked_species):
        SpeciesGraph(tracked_species).plot()

    def _gather_and_plot(
        self, gather_function, plot_function, return_data=False, **kwargs
    ):
        data = gather_function()
        plot_function(data, **kwargs)

        if return_data:
            return data

    def _visualize_toxin_model(self, data, separate_images):
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024**2)  # Convert bytes to MB
        end_mem = process.memory_info().rss / (1024**2)
        print(f"Memory used: {end_mem - start_mem:.2f} MB")

        v = self.model_factory.visualizer(separate_images)
        parameters = self.model_factory.parameters.from_config(self.config)
        da = self.model_factory.data_analysis(data, parameters)

        expected_equilibrium_cooperators = da.expected_equilibrium_cooperators()
        expected_equilibrium_fitness = da.expected_equilibrium_fitness()

        self._gather_and_plot(
            gather_function=da.get_fitness_df,
            plot_function=v.plot_fitness,
            expected_equilibrium=expected_equilibrium_fitness,
        )

        self._gather_and_plot(
            gather_function=da.get_genes_df,
            plot_function=v.plot_genes,
            expected_equilibrium=expected_equilibrium_cooperators,
            max_y=parameters.n_agents,
        )

        self._gather_and_plot(
            gather_function=da.get_agents_activity_df,
            plot_function=v.plot_agents_activity,
            max_y=parameters.n_agents,
        )

        self._gather_and_plot(
            gather_function=da.get_toxin_df,
            plot_function=v.plot_toxin_concentrations,
            max_y=parameters.toxin_base,
        )

        self._gather_and_plot(
            gather_function=da.get_redundant_genes_df,
            plot_function=v.plot_redundant_genes,
        )

        self._gather_and_plot(
            gather_function=da.get_species_per_composition_df,
            plot_function=v.plot_species,
            max_y=parameters.n_agents,
            title="Individuals of species per composition",
            legend=True,
        )

        self._gather_and_plot(
            gather_function=da.get_species_per_generation_df,
            plot_function=v.plot_species,
            col=1,
            title="Species population size per generation",
            legend=True,
        )

        self._gather_and_plot(
            gather_function=da.get_number_of_species_per_generation_df,
            plot_function=v.plot_n_species_per_generation,
            row=0,
            col=2,
        )

        self._gather_and_plot(
            gather_function=da.get_species_fitness_df,
            plot_function=v.plot_species_fitness,
            legend=True,
        )

        self._gather_and_plot(
            gather_function=da.get_species_representation_coefficient_df,
            plot_function=v.plot_species_composition_representation,
            row=0,
            col=2,
            legend=True,
        )

        self._gather_and_plot(
            gather_function=da.get_composition_diversity_fitness_correlation_df,
            plot_function=v.plot_diversity_fitness_correlation,
        )

        v.show_config(self.config)

        end_mem = process.memory_info().rss / (1024**2)
        print(f"Memory used: {end_mem - start_mem:.2f} MB")
        return v

    def _visualize_predator_prey(self, data, separate_images):
        v = self.model_factory.visualizer(separate_images)
        parameters = self.model_factory.parameters.from_config(self.config)
        da = self.model_factory.data_analysis(data, parameters)

        self._gather_and_plot(
            gather_function=da.get_fitness_df, plot_function=v.plot_fitness
        )

        self._gather_and_plot(
            gather_function=da.get_species_per_composition_df,
            plot_function=v.plot_species,
            max_y=parameters.n_predators,
            title="Species per composition",
            legend=True,
        )

        self._gather_and_plot(
            gather_function=da.get_species_per_generation_df,
            plot_function=v.plot_species,
            col=1,
            title="Species per generation",
            legend=True,
        )

        self._gather_and_plot(
            gather_function=da.get_number_of_species_per_generation_df,
            plot_function=v.plot_n_species_per_generation,
            row=0,
            col=1,
        )

        self._gather_and_plot(
            gather_function=da.get_species_fitness_df,
            plot_function=v.plot_species_fitness,
            legend=True,
        )

        self._gather_and_plot(
            gather_function=da.get_species_representation_coefficient_df,
            plot_function=v.plot_species_composition_representation,
            row=0,
            col=2,
            legend=True,
        )

        self._gather_and_plot(
            gather_function=da.get_composition_diversity_fitness_correlation_df,
            plot_function=v.plot_diversity_fitness_correlation,
        )

        self._gather_and_plot(
            gather_function=da.get_benchmark_df, plot_function=v.plot_benchmark
        )

        self._gather_and_plot(
            gather_function=da.get_catches_per_species_df,
            plot_function=v.plot_catches_per_species,
            legend=True,
        )

        chromosomes_tsne_df = self._gather_and_plot(
            gather_function=lambda: da.get_chromosomes_cluster_df(
                cluster_method="t-sne"
            ),
            plot_function=v.plot_chromosomes_tsne,
            row=2,
            col=1,
            return_data=True,  # Assuming this flag makes _gather_and_plot return the data
        )
        v.plot_fitness_landscape(chromosomes_tsne_df, row=0, col=1)

        self._gather_and_plot(
            gather_function=lambda: da.get_chromosomes_cluster_df(
                cluster_method="umap"
            ),
            plot_function=v.plot_chromosomes_umap,
            row=2,
            col=1,
        )
        v.show_config(self.config)

        return v

    def _visualize_function_optimization(self, data, separate_images):
        v = self.model_factory.visualizer(separate_images)
        parameters = self.model_factory.parameters.from_config(self.config)
        da = self.model_factory.data_analysis(data, parameters)

        self._gather_and_plot(
            da.average_objective_value_per_generation, v.plot_objective_value_over_time
        )

        self._gather_and_plot(
            da.get_solution_vector_and_objective_values,
            v.plot_solution_vector_to_objective_value,
        )

        self._gather_and_plot(da.get_fitness_df, v.plot_fitness, row=1)

        self._gather_and_plot(
            da.get_species_per_composition_df,
            v.plot_species,
            title="Species per composition",
            legend=True,
        )

        self._gather_and_plot(
            da.get_species_per_generation_df,
            v.plot_species,
            col=1,
            title="Species per generation",
            legend=True,
        )

        self._gather_and_plot(
            da.get_number_of_species_per_generation_df,
            v.plot_n_species_per_generation,
            col=1,
        )

        self._gather_and_plot(
            da.get_species_fitness_df, v.plot_species_fitness, legend=True
        )

        self._gather_and_plot(
            da.get_species_representation_coefficient_df,
            v.plot_species_composition_representation,
            legend=True,
        )

        self._gather_and_plot(
            da.get_composition_diversity_fitness_correlation_df,
            v.plot_diversity_fitness_correlation,
        )

        self._gather_and_plot(
            da.get_nonzero_gene_count_per_chromosome_df, v.plot_nonzero_genes
        )

        self._gather_and_plot(da.get_composition_size_df, v.plot_composition_size)

        # v.plot_value_over_time(best_value_per_generation, row=1)
        # v.plot_value_over_time(worst_value_per_generation, row=1)
        v.show_config(self.config)
        return v
