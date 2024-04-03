import matplotlib.pyplot as plt
import pandas as pd
from shared_components.model_visualizer import ModelVisualizer


class ToxinVisualizer(ModelVisualizer):
    def __init__(self, separate_plots=False):
        super().__init__(4, 3, separate_plots=separate_plots)
        self.rolling_windows = 10
        self.linewidth = 2
        if not self.separate_plots:
            self.fig.set_size_inches(14.5, 8.5)
            self.fig.suptitle("Toxin model", fontsize=16, x=0.75, y=0.51)

    def plot_genes(
        self,
        genes_per_step: pd.DataFrame,
        expected_equilibrium=None,
        max_y=None,
        row=0,
        col=1,
    ):
        """Plot the genes of the agents in the model."""
        self.plot_lineplot(
            df=genes_per_step,
            yline=expected_equilibrium,
            row=row,
            col=col,
            min_y=0,
            max_y=max_y * 1.1 if max_y != None else None,
            title="Gene frequency",
            xlabel="Generation",
            ylabel="Gene count",
            label_prefix="",
            legend_title="Gene",
        )

    def plot_agents_activity(
        self, agent_activity: pd.DataFrame, max_y=None, row=1, col=0
    ):
        # plot number of active genes per agent over time.
        # -> a line for each number of active genes, with y equalled to the number of agents with that many genes active
        self.plot_lineplot(
            df=agent_activity,
            row=row,
            col=col,
            min_y=0,
            max_y=max_y,
            title="Active functions",
            xlabel="Generation",
            ylabel="Agent count",
            label_suffix="",
            legend_title="Active functions",
        )

    def plot_toxin_concentrations(
        self, toxin_concentrations: pd.DataFrame, max_y=None, row=1, col=1
    ):
        """Plot the toxin concentrations in the environment."""
        self.plot_lineplot(
            df=toxin_concentrations,
            row=row,
            col=col,
            min_y=0,
            max_y=max_y * 1.1 if max_y != None else None,
            title="Toxin levels",
            xlabel="Generation",
            ylabel="Toxin level",
            label_prefix="",
            legend_title="Toxin",
        )

    def plot_redundant_genes(
        self, redundant_genes_per_step: pd.DataFrame, max_y=None, row=2, col=1
    ):
        """Plot the genes of the agents in the model."""
        self.plot_lineplot(
            df=redundant_genes_per_step,
            row=row,
            col=col,
            min_y=0,
            max_y=max_y,
            title="Redundant active functions",
            xlabel="Generation",
            ylabel="Gene count (past saturation)",
            label_prefix="",
            legend_title="Gene",
        )

    def show_config(self, c, row=2, col=2):
        model_parameters = [
            "N_RUNS",
            "N_GENERATIONS",
            "N_COMPOSITIONS",
            "N_AGENTS",
            "N_TOXINS",
            "TOXIN_BASE",
            "TOXIN_CLEANUP_RATE",
            "GENE_COST",
            "GENE_COST_MULTIPLIER",
            "EVOLUTION_TYPE",
            "CHROMOSOME_SAMPLING",
            "DETECT_SPECIES_METHOD",
            "COLLECTIVE_FITNESS_WEIGHT",
            "SCALE_FITNESS_WITH_REPRESENTATION",
            "ALIGN_COMPOSITIONS_WITH_POPULATION",
            "COLLECTIVE_FITNESS_TYPE",
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

        e_parameters["Composition"] = []
        if c["CHROMOSOME_SAMPLING"] == "evolutionary":
            e_parameters["Composition"] = [
                ("", c["COMPOSITION"][parameter])
                if parameter in c["COMPOSITION"]
                else ("", "")
                for parameter in evolutionary_parameters
            ]
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
            y_offset=-0.1,
            config_index=2,
        )
