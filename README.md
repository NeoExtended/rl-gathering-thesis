# Baselines Lab
Baselines lab is a experimentation environment for Reinforcement Learning built on top of [Stable-Baselines](https://github.com/hill-a/stable-baselines).
It offers easy configuration and deployment of RL algorithms via yaml or json config files in a simple syntax and inspired by [SLMLab](https://github.com/kengz/SLM-Lab).

## About
This project was created in 2020 as part of my master thesis to study the application of Reinforcement Learning on the targeted drug delivery problem.
In addition to the code, this repository contains the complete thesis, including the experimental results as well as a detailed documentation of the test environment.

## Installation
Currently Baselines Lab cannot be installed via pip and can only be obtained by cloning this repository.

Baselines Lab requires a number of Python libraries. For the installation of Stable-Baselines please refer to the [documentation](https://stable-baselines.readthedocs.io/en/master/guide/install.html).
Everything else can be installed using pip or conda via

```bash
pip install numpy opencv-python matplotlib imageio gym optuna pandas Pillow PyYAML
```

## Usage Example
The lab contains a run script at /baselines_lab/run_lab.py which can be used to start a session.
To train a simple PPO agent for the CartPole environment create a new configuration file [cartpole.yml](/config/exaples/cartpole.yml)

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


