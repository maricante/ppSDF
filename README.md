# ppSDF

![reconstructions_alpha](https://github.com/maricante/ppSDF/assets/13221985/ad930a4b-7412-4ee1-903a-bcb3bd8c454b)
This repository contains code examples for [*Marić et al.: Online learning of Continouous Signed Distance Fields Using Piecewise Polynomials*](https://sites.google.com/view/pp-sdf/).

This repository is a fork of [the RDF codebase](https://github.com/yimingli1998/RDF).

## Dependencies

Tested on *Python 3.10.12*

To install dependencies, run:

`pip install -r requirements.txt`

## Conda setup (tested)

To setup a conda environment, run:

`conda create --name ppSDF python=3.10.12`,

and install the required packages:

`python3 -m pip install -r requirements.txt`.

On Ubuntu, you might also need to install the GNU C++ runtime library to run the visualization:

`conda install -c conda-forge libstdcxx-n`

## Downloading the YCB dataset

The desired objects from the [YCB dataset (*Calli et al.*)](http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/) can be downloaded by running `python ycb_downloader.py`

Objects can be selected by commenting/uncommenting elements in the `objects_to_download` list inside the script.

## Running the example script

To run the example script with the desired arguments:

- `n_seg` - number of segments per input dimension
- `qd`, `qn`, `qt` - cost coefficients for incremental learning
- `sigma` - measurement noise
- `batch_size` - batch size for the incremental updates
- `device` - device to run on (e.g., `cuda` or `cpu`)
- `save` - if `True`, saves the approximated SDF
- `object` - which object to load
- `n_data` - number of training sample points
- `cut_x`, `cut_y`, `cut_z` - select axis to cut for visualization

For example, to approximate the '035_power_drill' object with 5 segments per input dimension:

`python reconstruct.py --n_seg=5 --object=035_power_drill`

To run with default arguments:

`python reconstruct.py`

Maintained by Ante MARIĆ and licensed under the MIT License.

Copyright (c) 2024 Idiap Research Institute, https://idiap.ch/
