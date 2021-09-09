# HyperNeRF: A Higher-Dimensional Representation for Topologically Varying Neural Radiance Fields

This is the code for "HyperNeRF: A Higher-Dimensional Representation for Topologically Varying Neural Radiance Fields".

* [Project Page](https://hypernerf.github.io)
* [Paper](https://arxiv.org/abs/2106.13228)
* [Video](https://www.youtube.com/watch?v=qzgdE_ghkaI)

This codebase contains a re-implementation of Nerfies using [JAX](https://github.com/google/jax),
building on [JaxNeRF](https://github.com/google-research/google-research/tree/master/jaxnerf).
We have been careful to match implementation details and have reproduced the original
results presented in the paper.


## Setup
The code can be run under any environment with Python 3.8 and above.
(It may run with lower versions, but we have not tested it).

We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and setting up an environment:

    conda create --name hypernerf python=3.8

Next, install the required packages:

    pip install -r requirements.txt

Install the appropriate JAX distribution for your environment by  [following the instructions here](https://github.com/google/jax#installation). For example:

    # For CUDA version 11.1
    pip install --upgrade "jax[cuda111]" -f https://storage.googleapis.com/jax-releases/jax_releases.html


## Training
After preparing a dataset, you can train a Nerfie by running:

    export DATASET_PATH=/path/to/dataset
    export EXPERIMENT_PATH=/path/to/save/experiment/to
    python train.py \
        --base_folder $EXPERIMENT_PATH \
        --gin_bindings="data_dir='$DATASET_PATH'" \
        --gin_configs configs/test_local.gin

To plot telemetry to Tensorboard and render checkpoints on the fly, also
launch an evaluation job by running:

    python eval.py \
        --base_folder $EXPERIMENT_PATH \
        --gin_bindings="data_dir='$DATASET_PATH'" \
        --gin_configs configs/test_local.gin

The two jobs should use a mutually exclusive set of GPUs. This division allows the
training job to run without having to stop for evaluation.


## Configuration
* We use [Gin](https://github.com/google/gin-config) for configuration.
* We provide a couple preset configurations.
* Please refer to `config.py` for documentation on what each configuration does.
* Preset configs:
    - `hypernerf_vrig_ds_2d.gin`: The deformable surface configuration for the validation rig (novel-view synthesis) experiments.
    - `hypernerf_vrig_ap_2d.gin`: The axis-aligned plane configuration for the validation rig (novel-view synthesis) experiments.
    - `hypernerf_interp_ds_2d.gin`: The deformable surface configuration for the interpolation experiments.
    - `hypernerf_interp_ap_2d.gin`: The axis-aligned plane configuration for the interpolation experiments.


## Dataset
The dataset uses the [same format as Nerfies](https://github.com/google/nerfies#datasets).


## Citing
If you find our work useful, please consider citing:
```BibTeX
@article{park2021hypernerf
  author    = {Park, Keunhong and Sinha, Utkarsh and Hedman, Peter and Barron, Jonathan T. and Bouaziz, Sofien and Goldman, Dan B and Martin-Brualla, Ricardo and Seitz, Steven M.},
  title     = {HyperNeRF: A Higher-Dimensional Representation for Topologically Varying Neural Radiance Fields},
  journal   = {arXiv preprint arXiv:2106.13228},
  year      = {2021},
}
```
