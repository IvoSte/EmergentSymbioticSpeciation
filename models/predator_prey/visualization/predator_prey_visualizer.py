import matplotlib.pyplot as plt
import pandas as pd
from shared_components.model_visualizer import ModelVisualizer


class PredatorPreyVisualizer(ModelVisualizer):
    def __init__(self, separate_plots=False):
        super().__init__(separate_plots=separate_plots)
        self.rolling_windows = 10
        self.linewidth = 2
        if not self.separate_plots:
            self.fig.set_size_inches(14.5, 8.5)
            self.fig.suptitle("Predator prey model", fontsize=16, x=0.75, y=0.51)

    def plot_catches_per_species(
        self,
        df: pd.DataFrame,
        row=1,
        col=0,
        legend=True,
    ):
        self.plot_lineplot(
            df=df,
            row=row,
            col=col,
            title="Catches per species",
            xlabel="Generation",
            ylabel="Catches",
            label_prefix="",
            legend=legend,
            legend_title="Species",
            legend_entry_limit=6,
        )

    def show_config(self, c, row=2, col=2):
        model_parameters = [
            "N_RUNS",
            "RUNS_PER_SET",
            "N_GENERATIONS",
            "N_COMPOSITIONS",
            "N_AGENTS",
            "EVOLUTION_TYPE",
            "CHROMOSOME_SAMPLING",
            "DETECT_SPECIES_METHOD",
            "COLLECTIVE_FITNESS_WEIGHT",
            "INDIVIDUAL_FITNESS_TYPE",
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
        ]

        m_parameters = {
            "Model": [(parameter, "") for parameter in model_parameters],
        }
        m_parameters[""] = [("", c[parameter]) for parameter in model_parameters]

        if c["SPECIES_TRACKING"]:
            m_parameters["Species tracking"] = [
                (parameter, c[parameter])
                for parameter in [
                    "SPECIES_TRACKING",
                    "SPECIES_TRACKING_SEARCH_DEPTH",
                    "SPECIES_TRACKING_DISTANCE_THRESHOLD",
                    "SPECIES_TRACKING_PROTOTYPE_METHOD",
                ]
            ]

        e_parameters = {
            "Evolutionary": [(parameter, "") for parameter in evolutionary_parameters],
        }
        e_parameters["Agent"] = [
            ("", c["AGENT"][parameter]) for parameter in evolutionary_parameters
        ]

        e_parameters["Composition"] = []
        if c["CHROMOSOME_SAMPLING"] == "evolutionary":
            e_parameters["Composition"] = [
                ("", c["COMPOSITION"][parameter])
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
            column_x=[-0.3, 0.3, 0.4],
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
