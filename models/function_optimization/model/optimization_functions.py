# library for the optimization functions
import numpy as np


APPLY_OFFSETS = True

class BenchmarkFunction:
    def evaluate(self, solution):
        raise NotImplementedError

    def apply_offsets(self, solution):
        if APPLY_OFFSETS:
            solution = [value + index for index, value in enumerate(solution)]
        return solution

class Rastrigin(BenchmarkFunction):
    def evaluate(self, solution):
        solution = self.apply_offsets(solution)
        n = len(solution)
        return 10 * n + sum([(x**2 - 10 * np.cos(2 * np.pi * x)) for x in solution])


class Ackley(BenchmarkFunction):
    def evaluate(self, solution):
        solution = self.apply_offsets(solution)
        n = len(solution)
        sum1 = sum([x**2 for x in solution])
        sum2 = sum([np.cos(2 * np.pi * x) for x in solution])
        return -20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e


class Schwefel(BenchmarkFunction):
    def evaluate(self, solution):
        solution = self.apply_offsets(solution)
        return 418.9829 * len(solution) - sum(
            [x * np.sin(np.sqrt(abs(x))) for x in solution]
        )


class Griewangk(BenchmarkFunction):
    def evaluate(self, solution):
        solution = self.apply_offsets(solution)
        sum_part = sum([x**2 / 4000.0 for x in solution])
        prod_part = np.prod(
            [np.cos(x / np.sqrt(i + 1)) for i, x in enumerate(solution)]
        )
        return 1 + sum_part - prod_part


class Sphere(BenchmarkFunction):
    def evaluate(self, solution):
        solution = self.apply_offsets(solution)
        return sum([x**2 for x in solution])


class Sum(BenchmarkFunction):
    def evaluate(self, solution):
        solution = self.apply_offsets(solution)
        return sum(abs(x) for x in solution)


class SumOffset(BenchmarkFunction):
    def evaluate(self, solution):
        solution = self.apply_offsets(solution)
        return sum(abs(x) for x in solution) + 100


class FullySeparable(BenchmarkFunction):
    def evaluate(self, solution):
        solution = self.apply_offsets(solution)
        offset = 50
        return sum([(x - offset) ** 2 for x in solution])


class PartiallySeparable(BenchmarkFunction):
    def evaluate(self, solution):
        solution = self.apply_offsets(solution)
        offset_a = 5
        offset_b = 10
        n = len(solution)
        half_n = n // 2
        x_a = solution[:half_n]
        x_b = solution[half_n:]
        return sum([(x - offset_a) ** 2 for x in x_a]) + (sum(x_b) - offset_b) ** 2


class FullyNonSeparable(BenchmarkFunction):
    def evaluate(self, solution):
        solution = self.apply_offsets(solution)
        offset = 5
        return (sum(solution) - offset) ** 2


N_SUBCOMPONENTS = 5


class StronglyPartiallySeparable(BenchmarkFunction):
    # Define the subfunctions g_i
    def subfunction(self, sub_sol):
        # Example of a strongly coupled subfunction
        return np.tanh(np.sum(sub_sol)) ** 2

    def evaluate(self, solution):
        solution = self.apply_offsets(solution)
        solution = np.array(solution)
        # offsets = np.array(
        #     range(len(solution))
        # )  # can be adjusted, now goes to [0, 1, 2, ...]
        # adjusted_solution = solution - offsets

        # Split the solution into subcomponents
        # Assuming an even split for simplicity, can be modified as needed
        n_subcomponents = N_SUBCOMPONENTS  # number of subcomponents, can be adjusted
        sub_solutions = np.array_split(solution, n_subcomponents)

        # Calculate the product of subfunctions
        result = np.sum([self.subfunction(sub_sol) for sub_sol in sub_solutions])

        return result


class BenchmarkFunctionFactory:
    @staticmethod
    def create_function(name):
        if name == "Rastrigin":
            return Rastrigin()
        elif name == "Ackley":
            return Ackley()
        elif name == "Schwefel":
            return Schwefel()
        elif name == "Griewangk":
            return Griewangk()
        elif name == "Sphere":
            return Sphere()
        elif name == "Sum":
            return Sum()
        elif name == "SumOffset":
            return SumOffset()
        elif name == "FullySeparable":
            return FullySeparable()
        elif name == "PartiallySeparable":
            return PartiallySeparable()
        elif name == "FullyNonSeparable":
            return FullyNonSeparable()
        elif name == "StronglyPartiallySeparable":
            return StronglyPartiallySeparable()
        else:
            raise ValueError("Unknown function")
