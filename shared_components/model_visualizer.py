from .logger import log
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import os


class ModelVisualizer:
    def __init__(self, nrows=4, ncols=3, separate_plots=False):
        self.separate_plots = separate_plots
        self.figures = {}
        if not self.separate_plots:
            self.fig, self.axis = plt.subplots(nrows=nrows, ncols=ncols)
            self.fig.tight_layout()

    def show(self):
        if self.separate_plots:
            self.show_separate_plots()
        else:
            plt.show()

    def show_separate_plots(self):
        for title, fig in self.figures.items():
            plt.figure(fig.number)
            plt.suptitle(title)
            plt.show()

    def save(self, filepath):
        if self.separate_plots:
            self.save_separate_plots(filepath)
        else:
            self.fig.savefig(filepath)

    def save_separate_plots(self, filepath):
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        for title, fig in self.figures.items():
            fig.savefig(os.path.join(filepath, title))

    def get_ax(self, row, col, title, return_fig=False):
        if self.separate_plots:
            fig, ax = plt.subplots()
            assert (
                title not in self.figures
            ), "Model plot title conflict, two graphs with the same title."
            self.figures[title] = fig
        else:
            ax = self.axis[row][col]
            fig = self.fig
        if return_fig:
            return ax, fig
        else:
            return ax

    def plot_lineplot(
        self,
        df: pd.DataFrame,
        row=0,
        col=0,
        min_y=None,
        max_y=None,
        title="title",
        xlabel="xlabel",
        ylabel="ylabel",
        label_prefix="",
        label_suffix="",
        legend=True,
        legend_title=None,
        legend_entry_limit=None,
        yline=None,
    ):
        ax = self.get_ax(row, col, title)

        rolling_average_window = max(1, int(len(df) / self.rolling_windows))
        ax.plot(
            df.rolling(rolling_average_window, min_periods=1).mean(),
            linestyle="-",
            linewidth=self.linewidth,
        )
        if yline != None:
            ax.axhline(y=yline, color="black", linestyle="--")
        if legend:
            ax.legend(
                [
                    f"{label_prefix}{str(column).replace('_', ' ')}{label_suffix}"
                    for column in df.columns[:legend_entry_limit]
                ],
                draggable=True,
                title=legend_title,
            )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=min_y, top=max_y)
        ax.grid()

    def plot_scatterplot(
        self,
        df: pd.DataFrame,
        xcol: str,
        ycol: str,
        row=0,
        col=0,
        min_y=None,
        max_y=None,
        title="title",
        xlabel="xlabel",
        ylabel="ylabel",
        labels=None,
        legend=True,
        legend_entry_limit=None,
        yline=None,
        alpha=0.1,
    ):
        ax = self.get_ax(row, col, title)

        color_mapping = self.labels_to_color_mapping(labels) if labels else None
        colors = [color_mapping[label] for label in labels] if labels else None

        ax.scatter(df[xcol], df[ycol], alpha=alpha, c=colors)

        if yline != None:
            ax.axhline(y=yline, color="black", linestyle="--")
        if legend:
            legend_handles = []
            for label, color in color_mapping.items():
                legend_handles.append(patches.Patch(color=color, label=str(label)))
            ax.legend(handles=legend_handles[:legend_entry_limit], draggable=True)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=min_y, top=max_y)
        ax.grid()

    def plot_3D_scatterplot(
        self,
        df: pd.DataFrame,
        xcol: str,
        ycol: str,
        zcol: str,
        row=0,
        col=0,
        title="title",
        xlabel="xlabel",
        ylabel="ylabel",
        zlabel="zlabel",
        legend=True,
        legend_entry_limit=None,
        alpha=0.1,
        labels=None,
        colors=None,
    ):
        ax, fig = self.get_ax(row, col, title, return_fig=True)

        ax.remove()  # Remove the 2D axis
        if not self.separate_plots:
            # ax.remove()  # Remove the 2D axis
            ax = fig.add_subplot(4, 3, row * 3 + col + 1, projection="3d")
        else:
            ax = fig.add_subplot(1, 1, 1, projection="3d")

        color_mapping = self.labels_to_color_mapping(labels) if labels else None
        colors = [color_mapping[label] for label in labels] if labels else None

        ax.scatter(
            df[xcol],
            df[ycol],
            df[zcol],
            alpha=alpha,
            c=colors,
        )
        if legend:
            legend_handles = []
            for label, color in color_mapping.items():
                legend_handles.append(patches.Patch(color=color, label=str(label)))
            ax.legend(handles=legend_handles[legend_entry_limit], draggable=True)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.grid()

    def plot_fitness(
        self,
        fitness_per_step: pd.DataFrame,
        expected_equilibrium=None,
        min_y=None,
        max_y=None,
        row=0,
        col=0,
        legend=True,
        legend_entry_limit=None,
    ):
        # """Plot the fitness of the agents in the model."""
        self.plot_lineplot(
            df=fitness_per_step,
            yline=expected_equilibrium,
            min_y=min_y,
            max_y=max_y,
            row=row,
            col=col,
            title="Average fitness",
            xlabel="Generation",
            ylabel="Fitness",
            legend=legend,
            legend_entry_limit=legend_entry_limit,
        )

    def plot_species(
        self,
        species_per_step: pd.DataFrame,
        row=3,
        col=0,
        max_y=None,
        title="Species",
        legend=True,
    ):
        """Plot the species in the model."""
        self.plot_lineplot(
            df=species_per_step,
            row=row,
            col=col,
            min_y=0,
            max_y=max_y,
            title=title,
            label_prefix="",
            xlabel="Generation",
            ylabel="Population size",
            legend=legend,
            legend_entry_limit=6,
            legend_title="Species",
        )

    def plot_n_species_per_generation(
        self,
        n_species_per_generation: pd.DataFrame,
        row=3,
        col=0,
        max_y=None,
        title="Number of species",
        legend=False,
    ):
        """Plot the species in the model."""
        self.plot_lineplot(
            df=n_species_per_generation,
            row=row,
            col=col,
            min_y=0,
            max_y=max_y,
            title=title,
            label_prefix="",
            xlabel="Generation",
            ylabel="Species count",
            legend=legend,
            legend_entry_limit=6,
            legend_title="Species",
        )

    def plot_species_fitness(
        self,
        species_fitness_per_step: pd.DataFrame,
        row=2,
        col=0,
        legend=True,
    ):
        """Plot the species in the model."""
        self.plot_lineplot(
            df=species_fitness_per_step,
            row=row,
            col=col,
            title="Species fitness",
            label_prefix="",
            xlabel="Generation",
            ylabel="Fitness",
            legend=legend,
            legend_title="Species",
            legend_entry_limit=6,
        )

    def plot_diversity_fitness_correlation(self, diversity_fitness_df, row=1, col=2):
        """Plot the correlation between diversity and fitness."""
        self.plot_scatterplot(
            df=diversity_fitness_df,
            xcol="shannon_diversity",
            ycol="collective_fitness",
            row=row,
            col=col,
            title="Diversity Fitness",
            xlabel="Composition Shannon diversity",
            ylabel="Composition fitness",
            alpha=0.1,
            legend=False,
        )

    def plot_species_composition_representation(
        self,
        species_per_step: pd.DataFrame,
        row=0,
        col=2,
        max_y=None,
        title="Species representation in compositions",
        legend=True,
    ):
        self.plot_lineplot(
            df=species_per_step,
            row=row,
            col=col,
            min_y=0,
            max_y=max_y,
            title=title,
            label_prefix="",
            xlabel="Generation",
            ylabel="Representation rate",
            legend=legend,
            legend_title="Species",
            legend_entry_limit=6,
        )

    def plot_benchmark(self, benchmark_df, row=1, col=1):
        """Plot the benchmark."""
        if len(benchmark_df) == 0:
            return
        self.plot_scatterplot(
            df=benchmark_df,
            xcol="generation_number",
            ycol="benchmark_score",
            row=row,
            col=col,
            min_y=0,
            max_y=10,
            title="Benchmark Score",
            xlabel="Generation",
            ylabel="Score",
            alpha=0.8,
            legend=False,
            yline=9,
        )

    def plot_chromosomes_tsne(self, chromosomes_tsne_df, row=1, col=0):
        """Plot the chromosomes in a 2D t-SNE scatterplot."""
        self.plot_scatterplot(
            df=chromosomes_tsne_df,
            xcol="x",
            ycol="y",
            row=row,
            col=col,
            title="Chromosomes t-SNE",
            xlabel="x",
            ylabel="y",
            alpha=0.9,
            legend=True,
            legend_entry_limit=6,
            labels=chromosomes_tsne_df["chromosome_type"].to_list(),
        )

    def plot_chromosomes_umap(self, chromosomes_umap_df, row=1, col=0):
        """Plot the chromosomes in a 2D UMAP scatterplot."""
        self.plot_scatterplot(
            df=chromosomes_umap_df,
            xcol="x",
            ycol="y",
            row=row,
            col=col,
            title="Chromosomes UMAP",
            xlabel="x",
            ylabel="y",
            alpha=0.9,
            legend=True,
            legend_entry_limit=6,
            labels=chromosomes_umap_df["chromosome_type"].to_list(),
        )

    def plot_fitness_landscape(self, clustered_chromosome_fitness_df, row=0, col=1):
        """Plot the fitness landscape of the chromosomes."""
        self.plot_3D_scatterplot(
            df=clustered_chromosome_fitness_df,
            xcol="x",
            ycol="y",
            zcol="fitness",
            row=row,
            col=col,
            title="Fitness landscape",
            xlabel="x",
            ylabel="y",
            zlabel="fitness",
            alpha=0.5,
            legend=False,
            legend_entry_limit=6,
            labels=clustered_chromosome_fitness_df["chromosome_type"].to_list(),
        )

    def labels_to_color_mapping(self, labels: list[str]):
        labels_set = sorted(set(labels), key=labels.index)
        if len(labels_set) <= 10:
            cmap = plt.colormaps["tab10"]
        elif len(labels_set) <= 20:
            cmap = plt.colormaps["tab20"]
        else:
            # Just because I like this map. It probably wont be used, just if there are too many species left.
            cmap = plt.colormaps["turbo"]

        color_list = cmap(range(len(labels_set)))
        color_mapping = {
            label: color_list[labels_set.index(label)] for label in labels_set
        }
        return color_mapping

    # def prepare_config(self, config, config_names):
    #     parameters = {
    #         parameter: {
    #             "display_name": parameter.lower().replace("_", " "),
    #             "config_value": config[parameter],
    #         }
    #         if parameter in config
    #         else {
    #             "display_name": parameter.lower().replace("_", " "),
    #             "config_value": "",
    #         }
    #         for parameter in config_names
    #     }
    #     return parameters

    def show_config(
        self,
        c=None,
        row=3,
        col=2,
        parameters=dict[list],
        column_x=[-0.3, 0.2, 0.8],
        y_offset=0.0,
        config_index=0,
    ):
        ax = self.get_ax(row, col, f"config_{config_index}")
        y_linehight = 0.1
        if self.separate_plots:
            column_x = [0.0, 0.4, 0.6]
            y_linehight = 0.05
        ax.axis("off")
        assert (
            type(parameters) == dict
        ), "Parameters should be a dict. Possibly the show config of the model visualizer is called directly and the child (domain model visualizer) class is not implemented."
        assert len(parameters) <= len(
            column_x
        ), "Too many parameter columns to show. Consider adding more column_x values."
        for x_idx, (parameter_title, parameter_list) in enumerate(parameters.items()):
            ax.text(
                column_x[x_idx],
                1.0 + y_offset,
                parameter_title,
                size=10,
                weight="bold",
                ha="left",
                va="center",
                transform=ax.transAxes,
            )
            for idx, parameter in enumerate(parameter_list):
                ax.text(
                    column_x[x_idx],
                    (1.0 + y_offset) - ((1 + idx) * y_linehight),
                    f"{parameter[0].lower()}: {parameter[1]}",
                    size=8,
                    ha="left",
                    va="center",
                    transform=ax.transAxes,
                )
