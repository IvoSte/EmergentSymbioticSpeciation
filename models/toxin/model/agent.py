from uuid import uuid4

GENE_ACTIVATION_THRESHOLD = 0.0  # warning: Changing this will require adjustment in binary species detection to function.


class Agent:
    def __init__(
        self,
        chromosome,
        environment,
        toxin_cleanup_rate=1,
        gene_cost=3,
        gene_cost_multiplier=1.0,
    ):
        self.id = uuid4()
        self.environment = environment
        self.chromosome = chromosome
        self.toxin_cleanup_rate = toxin_cleanup_rate
        self.gene_cost = gene_cost
        self.gene_cost_multiplier = gene_cost_multiplier

    @property
    def active_genes(self):
        return [
            idx
            for idx, gene in enumerate(self.chromosome.genes)
            if gene > GENE_ACTIVATION_THRESHOLD
        ]

    @property
    def inactive_genes(self):
        return [
            idx
            for idx, gene in enumerate(self.chromosome.genes)
            if gene <= GENE_ACTIVATION_THRESHOLD
        ]

    @property
    def gene_fitness_cost(self):
        return sum(
            [
                self.gene_cost * (self.gene_cost_multiplier**gene_idx)
                for gene_idx in range(len(self.active_genes))
            ]
        )

    @property
    def toxin_fitness_cost(self):
        return sum(
            [self.environment.toxins[gene_idx] for gene_idx in self.inactive_genes]
        )

    @property
    def fitness(self):
        fitness = -(self.toxin_fitness_cost + self.gene_fitness_cost)
        # print(f"self.gene_cost_multiplier: {self.gene_cost_multiplier}, self.gene_cost: {self.gene_cost}")
        # print(f"toxin_fitness_cost: {self.toxin_fitness_cost}, gene_fitness_cost: {self.gene_fitness_cost}, active_genes: {self.active_genes}")
        return fitness

    def step(self):
        for gene_idx in self.active_genes:
            self.environment.decrease_toxin(gene_idx, self.toxin_cleanup_rate)

    def get_data(self):
        return {
            "id": self.id,
            "chromosome": self.chromosome.genes,
            "chromosome_id": self.chromosome.chromosome_id,
            "chromosome_type": self.chromosome.chromosome_type,
            "active_genes": self.active_genes,
            "inactive_genes": self.inactive_genes,
            "individual_fitness": self.fitness,
            "gene_fitness_cost": -self.gene_fitness_cost,
            "toxin_fitness_cost": -self.toxin_fitness_cost,
        }
