from uuid import uuid4


class Agent:
    def __init__(self, chromosome, gene_cost=3, gene_cost_multiplier=1.0):
        self.id = uuid4()
        self.chromosome = chromosome
        self.values = self.chromosome_to_values(self.chromosome)
        self.objective_value = None
        self.gene_cost = gene_cost
        self.gene_cost_multiplier = gene_cost_multiplier

    def chromosome_to_values(self, chromosome):
        return chromosome.genes

    def set_objective_value(self, objective_value):
        self.objective_value = objective_value

    @property
    def _count_active_genes(self):
        return sum([1 for gene in self.chromosome.genes if gene != 0])

    @property
    def _gene_fitness_cost(self):
        return sum(
            [
                self.gene_cost * (self.gene_cost_multiplier**gene_idx)
                for gene_idx in range(self._count_active_genes)
            ]
        )

    @property
    def fitness(self):
        assert (
            self.objective_value is not None
        ), "Objective value must be set before calculating fitness"
        return -(self.objective_value + self._gene_fitness_cost)

    def get_data(self):
        data = {
            "id": self.id,
            "chromosome": self.chromosome.genes,
            "chromosome_id": self.chromosome.chromosome_id,
            "chromosome_type": self.chromosome.chromosome_type,
            "agent_values": self.values,
            "individual_fitness": self.fitness,
        }
        return data
