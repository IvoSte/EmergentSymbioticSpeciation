# EmergentSymbioticSpeciation
Master Thesis code repository.

Title: Emergent Symbiotic Speciation: Growing Holobionts with Cooperative Coevolution
Author: Ivo Steegstra

Abstract:
In nature, organisms larger than a microbe typically comprise organisms of multiple species. Species speciate, combine and co-evolve to perform different functional roles in an ecosystem. This thesis applies this biological framework to the field of computational evolution. We introduce a model featuring several novel mechanisms applied to the framework of cooperative coevolution, and it is shown that the model is capable of producing a composite solution of multiple emergent symbiotic species from a single gene pool, without prior configuration of the composition.
Speciation within a single gene pool is achieved through a novel assortative mating strategy. Emergent species are identified with unsupervised clustering, and solution compositions are formed by determining species type combinations with a genetic algorithm, thereby also facilitating meaningful coevolution. Finally a novel credit assignment strategy ensures survival of efficient altruism.
The model is evaluated with the behavioural dynamics produced by three distinct domain models: a predator-prey simulation, function optimization, and a novel 'Toxin' model designed to showcase the game theory dynamics inherent to the model.
Results show that dynamic emergent symbiotic speciation is achieved and produces good composite solutions. Species are not shown to be optimally decomposed subcomponents of the problem.

Note that this repository is the codebase of scientific research custom built for personal use.
It is made public in the spirit of open science and transparency.
The codebase is not intended for general use and is not maintained for that purpose.
This repository is a slightly altered cleaned up version of the original codebase, with some parts adjusted for usability.
However, you will still find plenty of comments to the programmer, commented out or redundant code, as is the nature of development code.

A large set of experiments, including the ones presented in the thesis, can be found in the ~/experiments/experiments directory.

## How to install the codebase

First, install python 3.11.2. Then, install the required packages by running the following command:

```bash
pip install -r requirements.txt
```

## How to run a domain model instance

To run an instance of a domain model, traverse to the directory of the domain model (models/_domain_model_name_).
Here you will find the configuration file config.toml in the config directory. Change parameters in this file to your liking.
Then, from the domain model directory, run the following command:

```bash
python main.py
```

## How to run the speciation model on a domain model

To perform a single run of the speciation model on a domain model, run the main.py file in the main directory. First, change the value of the ModelType instance argument in the main.py file to the desired domain model - ['predator_prey', 'toxin', 'function_optimization']. Adjust the configuration in the <model_name>_config.toml file in ~/config to adjust running settings. See the readme in the ~/config directory for specific instructions. 

Then, from the main directory run the following command:

```bash
python main.py
```

## How to run experiments

Performing a costum experiment with the speciation architecture can be done in two steps.

### 1. Create the experiment.
Create a new experiment configuration in the experiments directory. This this can be done most easily by generating an experiment set using the experiment generator, which can be found in the /experiments directory under the name 'experiment_generator.py'. In that python file in the main() function, configure the parameters to be changed for the experiment. See the documentation in that file for detailed instructions. The script uses a base config file as template, these can be found in the ~/experiments/experiments/_model_name_/base_configs, and which is used can be changed in the experiment_generator script. The experiment generator will create a new experiment set with a configuration file for each experiment. The experiment generator will create an experiment folder with the configurations in experiments/experiments/_model_name_/_experiment_name_. Here you will also find the results of the experiment after running.

After configuring, the experiment generator can be run with the following command:

```bash
python experiment_generator.py
```

### 2. Run the experiment.
Specific experiment running parameters can be changed in ~/config/experiment_runner_config.toml. Here you will find various output options.
Running the experiment is done with experiment_runner.py. This script will run the generated experiment set and can be run with the following command:

```bash
python experiment_runner.py -m <model_name> -e <experiment_name> 
```

where <model_name> is the name of the model to run ['predator_prey', 'toxin', 'function_optimization']the experiment on and <experiment_name> is the name of the generated experiment to run.

The experiment runner can be configured with the following commandline arguments:
    
```bash
-m, --model_name: The name of the model to run the experiment on.
-e, --experiment-set-name: The name of the experiment to run.
-o, --overwrite-existing-reports: If the experiment already exists, overwrite earlier results.
-r, --run-existing-experiments-again: If the experiment already exists, run the experiment again.
-mp, --multi-processing: Use multi-processing to run the experiments in parrallel.
-cpu, --max-num-cpus: The maximum number of CPUs to use. If not specified, all available CPUs will be used if multiprocessing is enabled.
```

E.G. to run the experiment set 'toxin_experiment_1' on the model 'toxin' with multi-processing on 4 cpus, use the following command:

```bash
python experiment_runner.py -m toxin -e toxin_experiment_1 -mp -cpu 4
```

To reiterate, the results can be found in the generated experiment folder in experiments/experiments/_model_name_/_experiment_name_.

To run several experiments, the 'run_several_experiments.sh' in ~/experiments can be used. Simply add the experiment_runner run commands to the file and run the file with the following command:

```bash
run_several_experiments.sh
```