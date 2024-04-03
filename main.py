# What is the purpose of this file?
# We want to run the model from the top level folder to use all speciation architecture
#
from shared_components.logger import log
from shared_components.config_loader import load_config
from speciation.model_runner import ModelRunner, ModelType


def main():
    model_type = ModelType(
        "predator_prey"
    )  # "predator_prey", "toxin" or "function_optimization"
    config = load_config(model_type)
    model_runner = ModelRunner(config)
    data = model_runner.run(verbose=True)
    model_runner.visualize_results(data)
    # model_runner.view_chromosome_benchmark_behaviour(loop=True)


def run_all_three_models():
    import matplotlib.pyplot as plt

    plt.ion()
    model_types = [
        ModelType.PREDATOR_PREY,
        ModelType.TOXIN,
        ModelType.FUNCTION_OPTIMIZATION,
    ]
    for model_type in model_types:
        config = load_config(model_type)
        model_runner = ModelRunner(config)
        data = model_runner.run(verbose=True)
        model_runner.visualize_results(data)

    input("Press enter to continue...")
    plt.ioff()


if __name__ == "__main__":
    main()
    # run_all_three_models()
