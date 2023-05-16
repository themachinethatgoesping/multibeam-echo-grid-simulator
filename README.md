# multibeam-echo-grid-simulator
This repository contains the simulation code (python) for reproducing the results of the accepted paper "Echo grid integration: A novel method for preprocessing multibeam water column data to quantify underwater gas bubble emissions." by Urban et. al. 2023 (Accepted) 

The program simulates multibeam water column images for virtual surveys over individual targets or target clouds and allows for testing the effects of different echo grid integration methods. (See Urban et. al 2023 for more details)

Note: the repository is currently beeing setup

# License
Mozilla Public License Version 2.0 (MPL 2.0)

# Installation
Currently you cannot install this simulation as a package. To run the code you have to add the src folder to you python path. This can be done from scripts.

#add mbes_sim to search path
import sys
sys.path.insert(0, "../src/")

import mbes_sim

You need the following packages with depencies from e.g. anaconda or pypi (version in brackets is the last tested versionls)

- python (3.10)
- numpy
- scipy
- numba
- pandas
- tqdm
- plotly
- jupyterlab
- ipympl


If you want to use the exact package versions that this simulation was last tested with, we recommend mambaforge (https://github.com/conda-forge/miniforge/releases/tag/23.1.0-1)

open the mamba promt and create a new environment with the packages specified in package-list.txt

mamba create -n mbes_sim --file package-list.txt
#activate the environment
mamba activate mbes_sim


# Examples
Work in progress
