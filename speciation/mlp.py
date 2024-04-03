import random
import math
import numpy as np
from dataclasses import dataclass
from .chromosome import Chromosome

INIT_WEIGHT_VAL = 0.5
INIT_BIAS_BAL = 0.5


def identity_activation(x):
    return x


def relu_activation(x):
    return max(0, x)


def sigmoid_activation(x):
    return 1 / (1 + math.exp(-x))


class MLPChromosome(Chromosome):
    def __init__(self, chromosome, n_input, n_hidden, n_output, hidden_layers):
        if chromosome == None:
            Chromosome.__init__(self)
        else:
            Chromosome.__init__(
                self,
                chromosome.genes,
                chromosome.chromosome_id,
                chromosome.chromosome_type,
            )
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.hidden_layers = hidden_layers

        self.n_input_weights = n_input * n_hidden
        self.n_hidden_weights = (hidden_layers - 1) * (n_hidden * n_hidden)
        self.n_output_weights = n_output * n_hidden

    @property
    def input_weights(self):
        input_weights = np.array(self.genes[: self.n_input_weights])
        input_weights.resize((self.n_input, self.n_hidden))
        return input_weights

    @property
    def hidden_weights(self):
        hidden_weights = np.array(
            self.genes[
                self.n_input_weights : self.n_input_weights + self.n_hidden_weights
            ]
        )
        hidden_weights.resize(self.hidden_layers - 1, self.n_hidden, self.n_hidden)
        return hidden_weights

    @property
    def output_weights(self):
        output_weights = np.array(
            self.genes[
                self.n_input_weights
                + self.n_hidden_weights : self.n_input_weights
                + self.n_hidden_weights
                + self.n_output_weights
            ]
        )
        output_weights.resize((self.n_hidden, self.n_output))
        return output_weights

    @property
    def biases(self):
        # If the chromosome has no biases, this returns an empty np array
        # Get the bias values from the chromosome tail
        flat_biases = self.genes[
            self.n_input_weights + self.n_hidden_weights + self.n_output_weights :
        ]
        cut_biases = []
        # Because there are a different amount of biases for hidden and the output layer,
        # we have to cut them carefully
        for hidden_layer in range(self.hidden_layers):
            cut_biases.append(
                np.array(
                    flat_biases[
                        hidden_layer
                        * self.n_hidden : (hidden_layer + 1)
                        * self.n_hidden
                    ]
                )
            )
        cut_biases.append(
            np.array(
                flat_biases[
                    self.hidden_layers
                    * self.n_hidden : (self.hidden_layers * self.n_hidden)
                    + self.n_output
                ]
            )
        )
        return np.array(cut_biases, dtype=object)


class MLP:
    def __init__(
        self,
        n_input,
        n_hidden,
        n_output,
        hidden_layers=1,
        activation_function=sigmoid_activation,
        output_activation_function=identity_activation,
        random_initialization=True,
        chromosome: Chromosome = None,
        bias=True,
    ):
        assert n_input > 0, "There should be at least one input node."
        assert n_hidden > 0, "There should be at least one hidden node."
        assert n_output > 0, "There should be at least one output node."
        assert hidden_layers > 0, "There should be at least one hidden layer."
        assert (
            type(chromosome) == Chromosome or chromosome == None
        ), "The input chromosome should be of type list, or None."

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.hidden_layers = hidden_layers

        self.bias = bias
        self.n_biases = bias * (n_hidden * hidden_layers + n_output)

        # position 0 corresponds to the weights connecting input 1 to hidden layer 1
        self.input_weights: numpy.ndarray = np.empty(
            (self.n_input, self.n_hidden), dtype=np.float16
        )
        # position [0][0][0] corresponds to the weights connecting hidden layer 1, node 1 to the next hidden layer node 1
        self.hidden_weights: numpy.ndarray = np.empty(
            (self.hidden_layers, self.n_hidden, self.n_hidden), dtype=np.float16
        )
        self.output_weights: numpy.ndarray = np.empty(
            (self.n_hidden, self.n_output), dtype=np.float16
        )
        self.biases: numpy.ndarray

        self.activation_function = np.vectorize(activation_function)
        self.output_activation_function = np.vectorize(output_activation_function)
        # Genes is a derived value, it is the amount of weights in the network total. All genes form a chromosome.
        self.n_genes = (
            (self.n_input * self.n_hidden)
            + ((self.n_hidden * self.n_hidden) * (self.hidden_layers - 1))
            + (self.n_hidden * self.n_output)
            + (self.n_biases)
        )
        self.chromosome = MLPChromosome(
            chromosome, n_input, n_hidden, n_output, hidden_layers
        )
        if chromosome == None:
            self.init_chromosome(self.n_genes, self.n_biases, random_initialization)

        self.init_weights_with_chromosome(self.chromosome)

    def init_chromosome(self, n_genes, n_biases, random_flag):
        if random_flag:
            self.chromosome.init_random(n_genes - n_biases)
            self.chromosome.add_genes_random(n_biases)
        else:
            self.chromosome.init_constant(n_genes - n_biases, INIT_WEIGHT_VAL)
            self.chromosome.add_genes_constant(n_biases, INIT_BIAS_BAL)

    def init_weights_with_chromosome(self, chromosome):
        assert (
            len(chromosome.genes) == self.n_genes
        ), f"Chromosome length {len(chromosome.genes)} does not match MLP specifications (i:{self.n_input} + {self.hidden_layers} * h:{self.n_hidden} + o:{self.n_output}) in MLP initialization."

        # Move through the flat chromosome and cut the weights from the genes -- the chromosome represents all weights in a 1D sequential list.
        self.input_weights = self.chromosome.input_weights
        self.hidden_weights = self.chromosome.hidden_weights
        self.output_weights = self.chromosome.output_weights
        self.biases = self.chromosome.biases

    def forward_pass(self, input_values):
        assert (
            len(input_values) == self.n_input
        ), f"Input length ({len(input_values)}) does not match number of input nodes ({self.n_input}) in input evaluation."

        input_values = np.array(input_values)
        activation_values = self.single_layer_forward_pass(
            input_values, self.input_weights, self.biases[0], self.activation_function
        )
        for layer_idx in range(self.hidden_layers - 1):
            activation_values = self.single_layer_forward_pass(
                activation_values,
                self.hidden_weights[layer_idx],
                self.biases[layer_idx + 1],
                self.activation_function,
            )

        output_values = self.single_layer_forward_pass(
            activation_values,
            self.output_weights,
            self.biases[-1],
            self.output_activation_function,
        )

        return output_values

    def single_layer_forward_pass(
        self, input_values, weights, bias, activation_function
    ):
        if self.bias:
            return activation_function(np.dot(input_values, weights) + bias)
        else:
            return activation_function(np.dot(input_values, weights))

    # def init_weights(self, random_flag):
    #     # Depricated, but function can still be used for faster initialization.
    #     if random_flag:
    #         self.input_weights = np.random.random(
    #             (self.n_input, self.n_hidden), dtype=np.float16
    #         )
    #         self.hidden_weights = np.random.random(
    #             (self.hidden_layers, self.n_hidden, self.n_hidden), dtype=np.float16
    #         )
    #         self.output_weights = np.random.random(
    #             (self.n_hidden, self.n_output), dtype=np.float16
    #         )
    #     else:
    #         self.input_weights = np.full(
    #             (self.n_input, self.n_hidden), INIT_WEIGHT_VAL, dtype=np.float16
    #         )
    #         self.hidden_weights = np.full(
    #             (self.hidden_layers - 1, self.n_hidden, self.n_hidden),
    #             INIT_WEIGHT_VAL,
    #             dtype=np.float16,
    #         )
    #         self.output_weights = np.full(
    #             (self.n_hidden, self.n_output),
    #             INIT_WEIGHT_VAL,
    #             dtype=np.float16,
    #         )

    def __str__(self):
        return f"Input:\n{self.input_weights}\nHidden:\n{self.hidden_weights}\nOutput:\n{self.output_weights}\nBiases:\n{self.biases}"


if __name__ == "__main__":
    mlp = MLP(
        n_input=2,
        n_hidden=4,
        n_output=2,
        hidden_layers=0,
        activation_function=sigmoid_activation,
        random_initialization=False,
        chromosome=None,
        bias=True,
    )

    # mlp.init_weights(False)
    print(mlp.forward_pass([1.0, 1.0]))
    print(mlp)
