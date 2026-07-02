# Code for simulations in "The emergence of a novel CA1 spatial map requires direct entorhinal input" (Chadney et al.) 

## Overview
This repository contains all the simulations and plots for the model part of the paper.

## Project Structure
* `run_simulation.py`: The main file containing the core simulations.
* `helpers.py`: Utility functions used across the notebooks/scripts.
* `neuron.py`: Main logic of the network of LIF neurons and plasticity.
* `analysis_and_plots.ipynb`: Here all the analyses of the simulated data are run and plots are created
* `plots/`: Folder containing the generated graphs.
* `results/`: Folder containing the statistical analyses of the results.


## Setup and Installation
To run this project locally, you will need Python installed. 

1. Clone the repository (if applicable).
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

## How to use

1. Run `run_simulation.py`. This will create all the simulations.
2. Run `analysis_and_plots.ipynb` to recreate the plots and results.