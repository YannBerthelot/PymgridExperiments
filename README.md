# PymgridExperiments

This Python repositery aims at implementing research experiments of Reinforcement Learning applied on the Pymgrid Environment of microgrids (cf: https://github.com/Total-RD/pymgrid) using custom made Deep RL agents (cf : https://github.com/YannBerthelot/DeepRL).

For stability reasons we use a modified version of pymgrid : https://github.com/YannBerthelot/pymgrid

This repo implements the following experiments :
-- Train an agent on the full year 
-- Train N agents over N clusters
-- Train an agent over fake data and test it over real data
-- Orchestrate agents trained over clusters to test them over a full year


## Installation

This packages requires deeprlyb and pymgrid. It uses poetry as a package manager
Packaging w.i.p.

```bash
git clone https://github.com/YannBerthelot/PymgridExperiments.git
cd PymgridExperiments
poetry install
poetry update
```

## Usage
e.g. : basic experiment. You have to define the experiment parameters in the config.ini file and then pass it as an argument to the execution.
```bash
poetry run python src/pymgridexperiments/pymgrid_experiment.py -s config.ini

```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)

