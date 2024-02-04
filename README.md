
# Skipper

A PyTorch Implementation of Skipper, proposed in the ICLR 2024 paper

**Consciousness-Inspired Spatio-Temporal Abstractions for Better Generalization in Reinforcement Learning**

-- *Mingde Zhao, Safa Alver, Harm van Seijen, Romain Laroche, Doina Precup, Yoshua Bengio*

[arXiv](https://arxiv.org/abs/2310.00229)

<a href="http://mingde.world/combining-spatial-and-temporal-abstraction-in-planning/" target="_blank">blogpost</a>

![skipper_cover](http://github.com/PwnerHarry/Skipper/assets/5063589/3a06bc2a-4b1d-4388-a1cd-cef6924c0451)

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

Please read carefully the args definition in `runtime.py` and pass the desired args in the commands.

## Extras
 - There is a potential CUDA_INDEX_ASSERTION error that could cause hanging at the beginning of the Skipper runs. We don't know yet how to fix it
 - The Dynamic Programming solutions for environment ground truth are only compatible with deterministic experiments
