# Research project on E(3) Equivariant Diffusion Model for Molecule Generation in 3D (EDM).

## Project Overview

### JAX Implementation
This repository introduces a JAX implementation of the E(3) Equivariant Diffusion Model (EDM), which was originally developed using PyTorch. By adapting this model to JAX, we leverage the framework's efficient execution and auto-differentiation capabilities to enhance both performance and scalability. JAX's support for GPU and TPU acceleration is particularly beneficial for handling the computationally intensive tasks involved in 3D molecular structure generation.

### Consistency Models
In a more significant departure from the original EDM diffusion, we also train a consistency 
model to speed up the sampling.

## Installation

To set up the environment for the E(3) Equivariant Diffusion Model in JAX, please refer to the `src/install_scripts/install.txt` file or `src/install_scripts/install_cpu.txt` if using cpu. This file contains all the necessary package requirements. Install these packages to ensure that the project runs smoothly.

To set up the environment for a consistency model, you can simply install the dependencies with  `pip install -r src/consistency_model_edm/requirements.txt`.

To run the project on the cluster, we also provide a def file to build a container as `src/install_scripts/singularity/jax_image.def`.

## Running the implementation

This project is currently in a testing phase, featuring a tested version of the E(3) Equivariant Diffusion Model that runs 
with limited parameters to verify its convergence and stability.


### Running the Tested Version
To observe how the model converges during this phase of development, execute the specific command provided below from the main directory `src/e3_diffusion_for_molecules-main` of the cloned repository:

`python main_qm9.py --exp_name exp_cond_alpha  --model egnn_dynamics --lr 1e-4  --nf 192 --n_layers 9 --save_model True --diffusion_steps 10 --sin_embedding False --n_epochs 10 --n_stability_samples 20 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --dequantization deterministic --include_charges False --diffusion_loss_type l2 --batch_size 64 --normalize_factors [1,8,1] --clip_grad False --conditioning alpha --dataset qm9_second_half --no_wandb`

### Training a consistency model

To train a consistency model, you can simply run `python src/consistency_model_edm/main_qm9.py` as is to start training. 
Naturally, the script support modifying parameters such as batch_size, learning rate, number of epochs, etc. with
flags, e.g. `python src/consistency_model_edm/main_qm9.py --batch_size=64 --n_epochs=10 --lr=1e-4`. 
