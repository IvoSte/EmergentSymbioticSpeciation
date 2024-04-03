import random


def

def generate_sets(n_sets, set_size):
    sets = []
    for idx in range(n_sets):
        sets.append([f"{chr(97+idx)}_{random.randint(1, 10)}" for _ in range(set_size[idx])])
    return sets

def generate_sets_alt(n_sets, n_items):
    # assign items to sets randomly
    pass

def check_desired_properties():
    # no combination collisions
    # item occurance approximately equal
    # type occurance approximately to proportion
    pass

def main():
    n_combinations = 10
    n_items = 30
    n_sets = 3

    #set_size = [random.randint(1, 10) for _ in range(n_sets)]
    sets = generate_sets(n_sets, set_size)
    print(sets)


if __name__ == "__main__":
    main()
