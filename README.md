
# Skipper

A PyTorch Implementation of Skipper, proposed in

  

**Combining Spatial and Temporal Abstraction in Planning for Better Generalization**

-- *Mingde Zhao, Safa Alver, Harm van Seijen, Romain Laroche, Doina Precup, Yoshua Bengio*

arXiv: pending

  

## Read carefully the args definition in runtime.py

  

## Python virtual environment configuration:

1. Create a virtual environment with conda or venv (we used Python 3.9)

2. Install PyTorch according to the official guidelines, make sure it recognizes your accelerators

3.  `pip install -r requirements.txt`

  

## For experiments, write bash scripts to call those Python files that start with string "run_":

`run_minigrid_mp.py`: a multi-processed experiment initializer for Skipper agents.

`run_minigrid.py`: a single-processed experiment initializer for modelfree baseline

`run_minigrid_with_CVAE.py`: a single-processed experiment initializer for training a checkpoint generator with the experience colleced by a modelfree or random baseline

`run_leap_pretrain_vae.py`: a single-processed experiment initializer for pretraining generator for the adapted LEAP agent

`run_leap_pretrain_rl.py`: a single-processed experiment initializer for pretraining distance estimator (policy) for the adapted LEAP agent

## Extras
 - There is a potential CUDA_INDEX_ASSERTION error that could cause hanging at the beginning of the Skipper runs. We don't know yet how to fix it
 - The Dynamic Programming solutions for environment ground truth are only compatible with deterministic experiments
