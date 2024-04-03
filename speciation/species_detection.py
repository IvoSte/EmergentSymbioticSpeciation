from sklearn.cluster import KMeans, DBSCAN, OPTICS
import numpy as np

from speciation.species_tracking import SpeciesTracker


class SpeciesDetector:
    """SpeciesDetector is a class that can be used to detect species in a population of genes.

    Args:
        method (str, optional): Method to use for detecting species. Defaults to "kmeans".
        calibrate_clusters_delta_range (int, optional): Range to calibrate the number of clusters for KMeans. Defaults to 1.
        eps (float, optional): Epsilon for DBSCAN and OPTICS. Defaults to 0.5.
        min_samples (int, optional): Minimum number of samples for DBSCAN and OPTICS. Defaults to 1.
        species_tracking (bool, optional): Whether to use species tracking -- use previous generations cluster prototypes to infer the most likely consistent species assignment.
        prototype_method (str, optional): Method to use for calculating the prototype of a set of genes. Defaults to "centroid".
    """

    def __init__(
        self,
        method: str = "kmeans",
        calibrate_clusters_delta_range: int = 1,
        eps: float = 0.5,
        min_samples: int = 1,
        species_tracking: bool = True,
        species_tracker: SpeciesTracker = None,
        prototype_method: str = "centroid",
    ):
        self.method = method
        self.calibrate_clusters_delta_range = calibrate_clusters_delta_range
        self.eps = eps
        self.min_samples = min_samples

        # species tracking
        self.species_tracking = species_tracking
        assert not (
            (species_tracking) and (species_tracker is None)
        ), "If species tracking is enabled, species tracker should not be None."
        self.prototype_method = prototype_method
        self.species_tracker = species_tracker
        # self.prototype_method = "centroid"
        # self.prototype_method = "density_peak"
        # self.prototype_method = "mediod"

    def get_quasispecies(
        self,
        genes,
        num_species=3,
        detect_species=False,
    ):
        # Top level function, only this one is called.
        np_genes = np.array(genes)

        if self.method == "kmeans":
            prediction = self.get_quasispecies_kmeans(
                np_genes, num_species, detect_species
            )
        elif self.method == "dbscan":
            prediction = self.get_quasispecies_dbscan(np_genes)
        elif self.method == "optics":
            prediction = self.get_quasispecies_optics(np_genes)
        elif self.method == "binary":
            prediction = self.get_quasispecies_binary(np_genes)
        elif self.method == "binary_knockout":
            prediction = self.get_quasispecies_binary_knockout(np_genes)
        else:
            raise ValueError(f"Method {self.method} not supported.")

        predicted_quasispecies = list(prediction)
        if self.species_tracking:
            genes_per_type = self._get_genes_per_type(genes, predicted_quasispecies)
            type_prototypes = self.get_quasispecies_prototypes(genes_per_type)
            type_cluster_size = {
                type_: len(type_genes) for type_, type_genes in genes_per_type.items()
            }

            species_translation_key = self.species_tracker.get_prototype_species(
                type_prototypes, type_cluster_size
            )
            predicted_quasispecies = [
                species_translation_key[type_] for type_ in predicted_quasispecies
            ]

        return predicted_quasispecies

    def get_quasispecies_dbscan(self, genes):
        prediction = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit_predict(
            genes
        )
        # d = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(genes)
        # prediction = d.labels_
        # print(f"core_sample_indices_ndarray: {d.core_sample_indices_}")
        # print(f"{d.get_params(deep = True)}")
        return prediction

    def get_quasispecies_optics(self, genes):
        prediction = OPTICS(eps=self.eps, min_samples=self.min_samples).fit_predict(
            genes
        )
        return prediction

    def get_quasispecies_kmeans(self, genes, num_species, detect_species=False):
        # if detect_species is true, we redetermine the number of species, by checking around the previous number of species.
        if detect_species:
            k_range = range(
                max(1, num_species - self.calibrate_clusters_delta_range),
                num_species + self.calibrate_clusters_delta_range + 1,
            )
            k = self.determine_k(genes, k_range)
        else:
            k = num_species

        prediction = KMeans(n_clusters=k, random_state=0, n_init="auto").fit_predict(
            genes
        )

        return prediction

    # def get_quasispecies_hierarchical_clustering(chromosmes):

    def get_quasispecies_binary(self, genes):
        # In the toxin model, the sign of the gene value is all that matters. Converting genes to binary values to integer gives us clear and easy species.
        binary_genes = [
            [1 if gene > 0 else 0 for gene in chromosome] for chromosome in genes
        ]
        prediction = [
            int(
                "".join(map(str, binary_chromosome)), 10
            )  # set 0 to 2 for base 2 int instead of bitstring
            for binary_chromosome in binary_genes
        ]
        return prediction

    def get_quasispecies_binary_knockout(self, genes):
        # In the toxin model, the sign of the gene value is all that matters. Converting genes to binary values to integer gives us clear and easy species.
        binary_genes = [
            [1 if gene != 0 else 0 for gene in chromosome] for chromosome in genes
        ]
        prediction = [
            int(
                "".join(map(str, binary_chromosome)), 10
            )  # set 0 to 2 for base 2 int instead of bitstring
            for binary_chromosome in binary_genes
        ]
        return prediction

    def determine_k(self, vectors, k_range) -> int:
        """Determine the number of clusters to use for the KMeans algorithm.

        Returns:
            The number of clusters to use for the KMeans algorithm.
        """
        best_bic = None
        best_k = 1
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(vectors)
            bic = self.calculate_bic_score(kmeans, vectors)
            # print(f"k: {k}, BIC: {bic}")
            if best_bic is None or bic < best_bic:
                best_bic = bic
                best_k = k
        return best_k

    def calculate_bic_score(self, kmeans, vectors):
        """Calculate the BIC score for a KMeans model."""
        # Number of data points and number of clusters
        n, k = vectors.shape[0], kmeans.n_clusters

        # Calculate the maximized value of the likelihood function
        L = -kmeans.score(vectors)

        # Calculate the BIC score
        bic = np.log(n) * k - 2 * np.log(L)
        return bic

    def get_quasispecies_prototypes(self, genes_per_type: dict):
        # get the prototypes of the set of genes.
        prototypes = {}
        for type_, type_genes_set in genes_per_type.items():
            prototypes[type_] = self.get_prototype(type_genes_set)
        return prototypes

    def _get_genes_per_type(self, genes: list[list[float]], prediction: list):
        # get the prototypes of the set of genes.
        genes_per_type = {}
        for genes_, type_ in zip(genes, prediction):
            if type_ not in genes_per_type:
                genes_per_type[type_] = []
            genes_per_type[type_].append(genes_)
        return genes_per_type

    def get_prototype(self, genes_set: list[list[float]]):
        # get the prototype of the set of genes.
        # if self.prototype_method == "centroid":
        prototype = self.get_centroid(genes_set)
        # elif self.prototype_method == "density_peak":
        #   prototype = self.get_density_peak(genes_set)
        # elif self,prototype_method == "mediod":
        #   prototype = self.get_mediod(genes_set)
        return prototype

    def get_centroid(self, genes_set: list[list[float]]):
        # get the centroid of the set of genes.
        genes_set_np = np.array(genes_set)
        centroid = np.mean(genes_set_np, axis=0)
        return centroid

    def get_density_peak(self, genes_set: list[list[float]]):
        # get the density peak of the set of genes.
        genes_set_np = np.array(genes_set)
        centroid = np.mean(genes_set_np, axis=0)
        return centroid

    def genomic_distance(self, genes_1: list[float], genes_2: list[float]):
        # calculate the genomic distance between two genes.
        genes_1_np = np.array(genes_1)
        genes_2_np = np.array(genes_2)
        distance = np.linalg.norm(genes_1_np - genes_2_np)
        return distance

    def species_detection_test(self, chromosomes):
        # THIS FUNCTION IS USED TO TEST THE PERFORMANCE OF THE SPECIES DETECTION ALGORITHMS.
        # The input are chromosomes with already assigned species.
        # It prints a comparison of the predicted species with the actual species.
        genes = [chromosome.genes for chromosome in chromosomes]

        # DO NOT REMOVE COMMENTED OUT CODE, IT IS USED FOR TESTING.
        # self.method = "kmeans"
        # genes_prediction_kmeans = self.get_quasispecies(
        #     genes,
        #     num_species=3,
        #     detect_species=True,
        # )
        # self.compare_species_with_prediction(
        #     chromosomes, genes_prediction_kmeans, "kmeans"
        # )

        self.method = "dbscan"
        genes_prediction_dbscan = self.get_quasispecies(genes)
        self.compare_species_with_prediction(
            chromosomes, genes_prediction_dbscan, "dbscan"
        )

        # self.method = "optics"
        # genes_prediction_optics = self.get_quasispecies(genes)
        # self.compare_species_with_prediction(
        #     chromosomes, genes_prediction_optics, "optics"
        # )

    def compare_species_with_prediction(self, chromosomes, genes_prediction, method):
        translation_counts = {}
        correct_types_set = set()
        inferred_types_set = set()
        for idx in range(len(chromosomes)):
            correct_type = chromosomes[idx].chromosome_type
            inferred_type = genes_prediction[idx]
            correct_types_set.add(correct_type)
            inferred_types_set.add(inferred_type)
            if correct_type not in translation_counts:
                translation_counts[correct_type] = {}
            if inferred_type not in translation_counts[correct_type]:
                translation_counts[correct_type][inferred_type] = 0
            translation_counts[correct_type][inferred_type] += 1

        translation_purity = {}
        for correct_type, inferred_types in translation_counts.items():
            translation_purity[correct_type] = max(inferred_types.values()) / sum(
                inferred_types.values()
            )
        print("\n")
        print(f"Species detection test using {method} method.")
        import pprint

        print(
            f"Number of chromosomes: {len(chromosomes)}\nNumber of predicted genes: {len(genes_prediction)}"
        )
        print(f"Correct species set: {correct_types_set}")
        print(f"Inferred species set: {inferred_types_set}")
        pp = pprint.PrettyPrinter(indent=4)
        print("Translation counts: [type]: [inferred type]: count")
        pp.pprint(translation_counts)
        print("Translation purity: [type]: purity")
        pp.pprint(translation_purity)
