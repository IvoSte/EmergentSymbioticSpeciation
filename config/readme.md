The configs in this directory are the main use and master configs of the different models, when running them using main.py.
Keep the master configs as is to store the default values of the different models.
Adjust the model_name_config.toml files to your needs, when running the models. The model will use the configs of {model_name}_config.toml named files.
Since this is a research project, not all configs will work out of the box.
Running a different model using main.py requires changing the string value of the model_type variable in main.py to the name of the model you want to run.

The configs for running the model as experiment (multiple runs, variations of configs) are generated (using experiment_generator.py) and stored in the experiments directory.
Running the experiments can be done using experiment_runner.py. Check the readme's and comments in the relevant files for more detailed instructions.