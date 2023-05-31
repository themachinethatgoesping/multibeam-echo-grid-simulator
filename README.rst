.. SPDX-FileCopyrightText: 2022 - 2023 Peter Urban, Ghent University
..
.. SPDX-License-Identifier: MPL-2.0

Welcome to the multibeam echo grid simulator!
=============================================
**multibeam echo grid simulator** (short: mbes_sim) simulates multibeam water column images for virtual surveys over individual targets or target clouds and allows for testing the effects of different echo grid integration methods. (See Urban et. al 2023 for more details)

The main purpose for this repository is to publish the code for reproducing the results of the accepted paper "Echo grid integration: A novel method for preprocessing multibeam water column data to quantify underwater gas bubble emissions." by Urban et. al. 2023 (Accepted) 

Usage / Examples
================
Please follow the :ref:`Installation` instructions bellow to create a python envornment using pip env or anaconda.

You will find several jupyter notebook examples in the 'examples' folder to:
- 01 Simulate for a survey over a single target (and visualize the resuts in 3D)
- 02 Simulate for a survey over a bubble stream (and visualize the resuts in 3D)
- 03 Simulate a single setup 10th or 100ths of times for statistical analyzis of this setup
- 04 Batch simulate multiple survey setups
- 05 Evaluate / plot simulation results

Commenting on the examples is currently mediocre (much worse than the actual simulation code). 

If you run into problems or have questions, please contact me: peter.urban@ugent.be.

License
=======
The code is distributed under the Mozilla Public License Version 2.0 (MPL-2.0)

In simple terms: The MPL-2.0 license implements a non-viral copyleft; Licensed files are protected by the copyleft, but they can still be deeply integrated even in comercial, closed source projects, as long as the file itself stays open source. 

Note that this simplified description is not a legal advice and does not cover all aspects of the license. For this please refer to the license self: https://www.mozilla.org/en-US/MPL/2.0/FAQ/

For other sources that may be easyer to comprehend see also

- https://www.mozilla.org/en-US/MPL/2.0/
- https://fossa.com/blog/open-source-software-licenses-101-mozilla-public-license-2-0/
- https://opensource.org/license/mpl-2-0/

Installation
============
Currently you cannot install this simulation as a package. To run the code you have to add the src folder to you python path. This can be done from scripts.

.. code-block:: python

  #add mbes_sim to search path
  import sys
  sys.path.insert(0, "../src/")

  import mbes_sim

You need the following packages with depencies from e.g. anaconda or pypi (version in brackets is the last tested versionls)

.. code-block:: python

  - python (tested with 3.10 and 3.11)
  - numpy
  - scipy
  - numba
  - pandas
  - pytables
  - tqdm
  - plotly
  - jupyterlab
  - ipympl

If you want to use the exact package versions that this simulation was last tested with, we recommend :ref:`https://github.com/conda-forge/miniforge/releases/tag/23.1.0-1<mambaforge>`

.. code-block:: python

  mamba create -n mbes_sim --file package-list.txt
  #activate the environment
  mamba activate mbes_sim
  
You can also use pip env

.. code-block:: python

  pipenv install -r requirements.txt
  
Or just install the required packages with with pip

.. code-block:: python

  pip install numpy scipy numba pandas pytables tqdm plotly jupyterlab ipympl


Contributing / Further development / Use
========================================

This project consists scripts created to reach results for a specific publication. While parts of the code are very well documented, other parts are not and can be more difficult to understand.

If you have questions, problems using the simulation or are interested in using/further developing the simulation code for another project, please contact me: peter.urban@ugent.be 
