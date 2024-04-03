from speciation.chromosome import Chromosome

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from matplotlib import colors as mcolors
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
from shared_components.logger import log


# Kind of depricated, still works, but mainly using the model plotter now.
class Visualizer:
    def __init__(self):
        self.colorlist = list(mcolors.CSS4_COLORS.keys())
        self.chromosome_visualizer = ChromosomeVisualizer()
        self.report_visualizer = ReportVisualizer()

    def visualize_chromosomes(
        self,
        chromosomes: list[Chromosome],
        technique: str,
        display: bool = False,
        save: bool = False,
        path: str = None,
        index: int = None,
    ) -> None:
        self.chromosome_visualizer.visualize_chromosomes(
            chromosomes, technique, display, save, path, index
        )

    def generate_plot(
        self,
        result_df: pd.DataFrame,
        plot_type: str,
        display: bool = False,
        save: bool = False,
        path: str = None,
        index: int = None,
    ) -> None:
        self.report_visualizer.visualize_report(
            result_df, plot_type, display, save, path, index
        )


class ReportVisualizer:
    def __init__(self) -> None:
        self.colorlist = list(mcolors.CSS4_COLORS.keys())

    def visualize_report(
        self,
        report_df: pd.DataFrame,
        plot_type: str,
        display: bool = False,
        save: bool = False,
        path: str = None,
        index: int = None,
    ) -> None:
        plt.figure()

        if plot_type == "benchmark":
            self.benchmark_score_plot(report_df)
        elif plot_type == "fitness":
            self.fitness_plot(report_df)
        elif plot_type == "species":
            self.species_plot(report_df)
        elif plot_type == "benchmark_aggregate":
            self.benchmark_aggregate_plot(report_df)
        elif plot_type == "fitness_aggregate":
            self.fitness_aggregate_plot(report_df)
        else:
            raise ValueError("Invalid plot type")

        if display:
            plt.show()

        if save:
            if path is None:
                path = os.path.join("img", "plot", plot_type)
            if not os.path.exists(path):
                os.makedirs(path)
            if index is None:
                filename = f"{plot_type}.png"
            else:
                filename = f"{plot_type}_{index}.png"
            plt.savefig(os.path.join(path, filename))
        plt.close()

    def benchmark_score_plot(self, benchmark_df: pd.DataFrame) -> None:
        df = benchmark_df[
            [
                "generation_number",
                "benchmark_score",
                "benchmark_best_combination_score",
                "benchmark_worst_combination_score",
            ]
        ]
        df.set_index("generation_number", inplace=True)

        plt.figure()
        plt.plot(df)
        plt.legend(df.columns)
        plt.title("Benchmark score")
        plt.xlabel("Generation")
        plt.ylabel("Benchmark score")
        plt.grid()

    def benchmark_best_combination(self, report_df: pd.DataFrame) -> None:
        pass

    def fitness_plot(self, fitness_df: pd.DataFrame) -> None:
        df = fitness_df

        df.set_index("generation_number", inplace=True)

        plt.figure()
        plt.plot(
            df[
                [
                    "mean_individual_fitness",
                    "mean_collective_fitness",
                    "mean_combined_fitness",
                ]
            ],
            linestyle="--",
        )
        plt.plot(
            df[["max_individual_fitness", "max_collective_fitness"]], linestyle=":"
        )
        plt.plot(
            df[["min_individual_fitness", "min_collective_fitness"]], linestyle="-."
        )
        plt.legend(df.columns)
        plt.title("Fitness")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.grid()

    def species_plot(self, species_df: pd.DataFrame) -> None:
        df = species_df
        df.set_index("generation_number")
        df = df.drop(columns=["generation_number"])
        plt.stackplot(df.index, df.values.T)
        # plt.legend(df.columns)
        plt.title("Species")
        plt.xlabel("Generation")
        plt.ylabel("Units per species")

    def benchmark_aggregate_plot(self, benchmark_aggregate_df: pd.DataFrame) -> None:
        df = benchmark_aggregate_df[
            [
                ("generation_number", "Unnamed: 1_level_1"),
                ("benchmark_score", "mean"),
                ("benchmark_score", "std"),
            ]
        ]
        df = df.fillna(0.0)
        plt.figure()
        plt.errorbar(
            x=df["generation_number"],
            y=df[("benchmark_score", "mean")],
            yerr=df[("benchmark_score", "std")],
            fmt="o",
        )
        plt.title("Aggregate benchmark")
        plt.xlabel("Generation")
        plt.ylabel("Benchmark score")
        plt.grid()

    def fitness_aggregate_plot(self, fitness_aggregate_df: pd.DataFrame) -> None:
        df = fitness_aggregate_df[
            [
                ("generation_number", "Unnamed: 1_level_1"),
                ("average_fitness_of_all", "mean"),
                ("average_fitness_of_all", "std"),
            ]
        ]
        df = df.fillna(0.0)
        plt.figure()
        plt.errorbar(
            x=df["generation_number"],
            y=df[("average_fitness_of_all", "mean")],
            yerr=df[("average_fitness_of_all", "std")],
            fmt="o",
        )
        plt.title("Aggregate fitness")
        plt.xlabel("Generation")
        plt.ylabel("Fitness score")
        plt.grid()


class ChromosomeVisualizer:
    def __init__(self):
        self.colorlist = list(mcolors.CSS4_COLORS.keys())

    def visualize_chromosomes(
        self,
        chromosomes: list[Chromosome],
        technique: str,
        display: bool = False,
        save: bool = False,
        path: str = None,
        index: int = None,
    ):
        genes = [chromosome.genes for chromosome in chromosomes]
        gene_types = [chromosome.chromosome_type for chromosome in chromosomes]

        # Create a numpy array of the genes.
        chromosome_df = pd.DataFrame(genes, index=gene_types)

        if chromosome_df.shape[0] <= 25 and technique != "heatmap":
            log.warning(
                f"Too few chromosomes to plot accurate 2D {technique} representation. Consider using > 25 chromosomes."
            )
            return

        plt.figure()

        if technique == "heatmap":
            self.chromosomes_heatmap(chromosome_df)
        elif technique == "t-sne":
            self.chromosomes_tsne(chromosome_df)
        elif technique == "pca":
            self.chromosomes_pca(chromosome_df)
        elif technique == "umap":
            self.chromosomes_umap(chromosome_df)

        plt.title(f"{technique}")
        if display:
            plt.show()

        if save:
            if path is None:
                path = os.path.join("img", technique)
            if not os.path.exists(path):
                os.makedirs(path)
            if index is None:
                index = 0
            plt.savefig(os.path.join(path, f"{technique}_{index}.png"))
        plt.close()

    def chromosomes_heatmap(self, chromosome_df):
        # Create a heatmap
        sns.heatmap(chromosome_df, cmap="YlGnBu")

    def chromosomes_tsne(self, chromosome_df):
        print(chromosome_df)
        # Create a color mapping
        color_map = {i: self.colorlist[i] for i in chromosome_df.index}
        colors = list(map(lambda x: color_map[x], chromosome_df.index))

        # Create a t-SNE plot
        tsne = TSNE(
            n_components=2,
            random_state=0,
            perplexity=25,
            init="random",
            learning_rate="auto",
        )
        X_2d = tsne.fit_transform(chromosome_df)

        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors)

    def chromosomes_umap(self, chromosome_df):
        # create color mapping
        color_map = {i: self.colorlist[i] for i in chromosome_df.index}
        colors = list(map(lambda x: color_map[x], chromosome_df.index))

        # Create a UMAP plot
        X_2d = UMAP(n_components=2, random_state=0).fit_transform(chromosome_df)

        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors)

    def chromosomes_pca(self, chromosome_df):
        # create color mapping
        color_map = {i: self.colorlist[i] for i in chromosome_df.index}
        colors = list(map(lambda x: color_map[x], chromosome_df.index))

        # Create a UMAP plot
        pca = PCA(n_components=2, random_state=0)
        X_2d = pca.fit_transform(chromosome_df)

        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors)
