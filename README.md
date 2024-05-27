# Research project on E(3) Equivariant Diffusion Model for Molecule Generation in 3D (EDM).

## Project Overview

### JAX Implementation
This repository introduces a JAX implementation of the E(3) Equivariant Diffusion Model (EDM), which was originally developed using PyTorch. By adapting this model to JAX, we leverage the framework's efficient execution and auto-differentiation capabilities to enhance both performance and scalability. JAX's support for GPU and TPU acceleration is particularly beneficial for handling the computationally intensive tasks involved in 3D molecular structure generation.

### Consistency Model Advancement
In a significant departure from the original model's diffusion-based approach, this implementation employs a consistency model to streamline molecular generation. This shift is aimed at accelerating the training and inference phases, effectively reducing computational demands while maintaining high fidelity in the molecular structures produced. The consistency model provides a robust alternative that optimizes the generation process, making it faster and more resource-efficient.
NOTE: This is still WIP and we decided not to publish code for this yet since it is currently a pytorch modification  to the original EDM repo and would interfere with the JAX code structure. However, we will add this by Thursday and all relevant theory and explanations are already in the blogpost.

## Installation

To set up the environment for the E(3) Equivariant Diffusion Model in JAX, please refer to the `src/install_scripts/install.txt` file or `src/install_scripts/install_cpu.txt` if using cpu. This file contains all the necessary package requirements. Install these packages to ensure that the project runs smoothly.

To run the project on the cluster, we also provide a def file to build a container as `src/install_scripts/singularity/jax_image.def`.

## Running the implementation

This project is currently in a testing phase, featuring a tested version of the E(3) Equivariant Diffusion Model that runs with limited parameters to verify its convergence and stability.

### Running the Tested Version
To observe how the model converges during this phase of development, execute the specific command provided below from the main directory `src/e3_diffusion_for_molecules-main` of the cloned repository:

`python main_qm9.py --exp_name exp_cond_alpha  --model egnn_dynamics --lr 1e-4  --nf 192 --n_layers 9 --save_model True --diffusion_steps 10 --sin_embedding False --n_epochs 10 --n_stability_samples 20 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --dequantization deterministic --include_charges False --diffusion_loss_type l2 --batch_size 64 --normalize_factors [1,8,1] --clip_grad False --conditioning alpha --dataset qm9_second_half --no_wandb`
