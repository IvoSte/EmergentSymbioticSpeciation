import matplotlib.pyplot as plt
import pandas as pd
from shared_components.model_visualizer import ModelVisualizer


class BenchmarkFunctionVisualizer(ModelVisualizer):
    def __init__(self, separate_plots=False):
        super().__init__(4, 3, separate_plots=separate_plots)
        self.rolling_windows = 10
        self.linewidth = 2
        if not self.separate_plots:
            self.fig.set_size_inches(14.5, 8.5)
            self.fig.suptitle(
                "Function optimization model", fontsize=16, x=0.75, y=0.51
            )

    def plot_objective_value_over_time(
        self, objective_value_per_generation: pd.DataFrame, row=0, col=0
    ):
        """Plot the value of the function over time."""
        self.plot_lineplot(
            df=objective_value_per_generation,
            row=row,
            col=col,
            min_y=0,
            title="Objective function value over time",
            xlabel="Generation",
            ylabel="Objective function value",
            legend=False,
        )

    def plot_solution_vector_to_objective_value(
        self, solution_value: pd.DataFrame, row=0, col=1
    ):
        """Visualize the solution vector to the objective value."""
        self.plot_3D_scatterplot(
            df=solution_value,
            xcol="solution_vector_x",
            ycol="solution_vector_y",
            zcol="objective_value",
            row=row,
            col=col,
            title="Solution values to objective function value",
            xlabel="Solution vector x",
            ylabel="Solution vector y",
            zlabel="Objective value",
            legend=False,
        )

    def plot_nonzero_genes(self, nonzero_genes: pd.DataFrame, row=1, col=1):
        """Plot the number of nonzero genes per chromosome."""
        self.plot_lineplot(
            df=nonzero_genes,
            row=row,
            col=col,
            min_y=0,
            title="Subcomponent sizes",
            label_prefix="",
            legend_title="subcomponent size",
            xlabel="Generation",
            ylabel="Subcomponent count",
        )

    def plot_composition_size(self, composition_size: pd.DataFrame, row=2, col=1):
        """Plot the size of the composition."""
        self.plot_lineplot(
            df=composition_size,
            row=row,
            col=col,
            min_y=0,
            title="Average composition size",
            xlabel="Generation",
            ylabel="Composition size",
            legend=False,
        )

    def show_config(self, c, row=2, col=2):
        model_parameters = [
            "FUNCTION_NAME",
            "N_RUNS",
            "N_GENERATIONS",
            "N_COMPOSITIONS",
            "N_AGENTS",
            "GENE_COST",
            "GENE_COST_MULTIPLIER",
            "EVOLUTION_TYPE",
            "CHROMOSOME_SAMPLING",
            "DETECT_SPECIES_METHOD",
            "COLLECTIVE_FITNESS_WEIGHT",
            "COLLECTIVE_FITNESS_TYPE",
            "SCALE_FITNESS_WITH_REPRESENTATION",
            "ALIGN_COMPOSITIONS_WITH_POPULATION",
        ]

        evolutionary_parameters = [
            "REPRODUCE_FRACTION",
            "CULL_FRACTION",
            "MUTATION_PROBABILITY",
            "MUTATION_RANGE",
            "SELECTION_TYPE",
            "MATE_SELECTION_TYPE",
            "MATE_SELECTION_SAMPLE_SIZE",
            "ALLOW_SELF_MATING",
            "KNOCKOUT_MUTATION",
            "KNOCKOUT_MUTATION_PROBABILITY",
            "CHROMOSOME_RESIZING",
            "CHROMOSOME_RESIZING_PROBABILITY",
        ]

        m_parameters = {
            "Model": [(parameter, "") for parameter in model_parameters],
        }
        m_parameters[""] = [("", c[parameter]) for parameter in model_parameters]

        e_parameters = {
            "Evolutionary": [(parameter, "") for parameter in evolutionary_parameters],
        }
        e_parameters["Agent"] = [
            ("", c["AGENT"][parameter]) if parameter in c["AGENT"] else ("", "")
            for parameter in evolutionary_parameters
        ]
        e_parameters["Agent"] = [
            ("", c["AGENT"][parameter]) if parameter in c["AGENT"] else ("", "")
            for parameter in evolutionary_parameters
        ]

        e_parameters["Composition"] = []
        if c["CHROMOSOME_SAMPLING"] == "evolutionary":
            e_parameters["Composition"] = [
                ("", c["COMPOSITION"][parameter])
                if parameter in c["COMPOSITION"]
                else ("", "")
                for parameter in evolutionary_parameters
            ]
        m_parameters["Model"].insert(1, ("N_DIMENSIONS", ""))
        m_parameters[""].insert(1, ("", c["N_GENES"]))
        e_parameters["Evolutionary"].insert(0, ("POPULATION_SIZE", ""))
        e_parameters["Agent"].insert(
            0, ("", c["N_SUBPOPULATIONS"] * c["SUBPOPULATION_SIZE"])
        )
        e_parameters["Composition"].insert(0, ("", c["N_COMPOSITIONS"]))
        super().show_config(
            c=c,
            row=row,
            col=col,
            parameters=m_parameters,
            column_x=[-0.3, 0.4],
            config_index=1,
        )
        super().show_config(
            c=c,
            row=row + 1,
            col=col,
            parameters=e_parameters,
            column_x=[-0.3, 0.4, 0.8],
            config_index=2,
        )
