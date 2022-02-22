# About
This repository contains the code and text for my Master's Thesis with the title "Reinforcement Learning for Navigating Particle Swarms by Global Force". 
The thesis contains an in-depth analysis of the application of RL on the particle gathering problem.

## Baselines Lab
Baselines lab was developed as part of my thesis and is an experimentation environment for Reinforcement Learning built on top of [Stable-Baselines](https://github.com/hill-a/stable-baselines).
It offers easy configuration and deployment of RL algorithms via yaml or json config files in a simple syntax and inspired by [SLMLab](https://github.com/kengz/SLM-Lab).

## Installation
Currently Baselines Lab cannot be installed via pip and can only be obtained by cloning this repository.

Baselines Lab requires a number of Python libraries. For the installation of Stable-Baselines please refer to the [documentation](https://stable-baselines.readthedocs.io/en/master/guide/install.html).
Everything else can be installed using pip or conda via

```bash
pip install numpy opencv-python matplotlib imageio gym optuna pandas Pillow PyYAML
```

## Usage Example
The lab contains a run script at /baselines_lab/run_lab.py which can be used to start a session.
To train a simple PPO agent for the CartPole environment create a new configuration file [cartpole.yml](/config/examples/cartpole.yml)

```yaml
algorithm:
  name: "ppo2"
  policy:
    name: "MlpPolicy"

env:
  name: "CartPole-v0"

meta:
  n_timesteps: 10000
```

Then run 

```bash
python run_lab.py train cartpole.yml
```

After training is done, you can replay your trained model using

```bash
python run_lab.py enjoy cartpole.yml
```


