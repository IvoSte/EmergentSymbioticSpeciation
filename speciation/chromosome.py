from uuid import uuid4
import random


class Chromosome:
    def __init__(self, genes=None, chromosome_id=None, chromosome_type="default_type"):
        self.genes: list = genes
        self.chromosome_id = chromosome_id if chromosome_id != None else uuid4()
        self.chromosome_type = chromosome_type

    def init_random(self, n_genes):
        self.genes = [random.random() - 0.5 for _ in range(n_genes)]

    def init_random_with_range(self, n_genes, min_value, max_value):
        self.genes = [random.uniform(min_value, max_value) for _ in range(n_genes)]

    def init_random_binary(self, n_genes):
        self.genes = [random.randint(0, 1) for _ in range(n_genes)]

    def init_constant(self, n_genes, constant):
        self.genes = [constant for _ in range(n_genes)]

    def add_genes_random(self, n_genes):
        self.genes += [random.random() - 0.5 for _ in range(n_genes)]

    def add_genes_constant(self, n_genes, constant):
        self.genes += [constant for _ in range(n_genes)]

    def knockout_genes(self, n_genes):
        knockout_indices = random.sample(range(len(self.genes)), n_genes)
        for index in knockout_indices:
            self.genes[index] = 0.0

    @staticmethod
    def generate_random(n_genes):
        chromosome = Chromosome()
        chromosome.init_random(n_genes)
        return chromosome

    @staticmethod
    def generate_random_with_range(n_genes, min_value, max_value):
        chromosome = Chromosome()
        chromosome.init_random_with_range(n_genes, min_value, max_value)
        return chromosome

    def __str__(self):
        return f"{self.chromosome_id = } {self.chromosome_type = }"
