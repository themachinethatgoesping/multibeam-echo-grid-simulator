# multibeam echo grid simulator
This repository contains the simulation code (python) for reproducing the results of the accepted paper "Echo grid integration: A novel method for preprocessing multibeam water column data to quantify underwater gas bubble emissions." by Urban et. al. 2023 (Accepted) 

The program simulates multibeam water column images for virtual surveys over individual targets or target clouds and allows for testing the effects of different echo grid integration methods. (See Urban et. al 2023 for more details)

# License
Mozilla Public License Version 2.0 (MPL-2.0)

In simple terms: The MPL-2.0 license implements a non-viral copyleft that only affects the licensed files; but not the project that include these files. MPL-2.0 licensed files can thus be deeply integrated even in comercial, closed source projects. Further, MPL-2.0 includes an explicit clause to make it compatible with GPL (>2.0) projects. 

Note that this simplified description is not a legal advice and does not cover all aspects of the license. For this please refer to the license self
- https://www.mozilla.org/en-US/MPL/2.0/FAQ/

For other sources that may be easyer to comprehend see also
- https://www.mozilla.org/en-US/MPL/2.0/
- https://fossa.com/blog/open-source-software-licenses-101-mozilla-public-license-2-0/
- https://opensource.org/license/mpl-2-0/

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

# Contributing / Further development / Use
This project represents scientific script created to reach results for a specific publication. General purpose usability was thus not a primary goal during the creation and there might be parts of the code that are difficult to understand.

If you have questions, problems using the simulation or are interested in using/further developing the simulation code for another project, please contact me: peter.urban@ugent.be 
