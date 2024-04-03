import pickle
from .chromosome import Chromosome
import os


def save_chromosome_set(chromosome_set: list[Chromosome], filepath: str = None):
    filepath = (
        filepath
        if filepath
        else os.path.join("data", "chromosomes", "chromosome_set.pkl")
    )
    with open(filepath, "wb") as f:
        pickle.dump(chromosome_set, f)


def load_chromosome_set(filepath: str = None):
    filepath = (
        filepath
        if filepath
        else os.path.join("data", "chromosomes", "chromosome_set.pkl")
    )
    with open(filepath, "rb") as f:
        return pickle.load(f)
