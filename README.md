![Install & Train](https://github.com/boettiger-lab/open_ai_fishing/workflows/Install%20&%20Train/badge.svg) [![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)

Here is the fishing gym. Work IN PROGRESS!

## Environments

This repository provides [OpenAI-gym](https://github.com/openai/gym/) class definitions for the fisheries management problem.  See [Creating your own Gym Environments](https://github.com/openai/gym/blob/master/docs/creating-environments.md) for details. 


Install the python `fishing_gym` module by cloning this repo and running:

```
python setup.py sdist bdist_wheel
pip install -e .
```

So far, we have:

Simple fishing model defined in a continuous state space of fish biomass, with:

- Discrete action space with three actions: maintain harvest level, increase harvest by 20%, decrease harvest by 20%
- Discrete action space with n > 3 actions: action is taken as quota, `quota = action / n_actions * K`
- Continuous action space, `action` = quota.

## Examples

Examples for running this gym environment in several frameworks:  

- [keras-rl](/keras-rl)
  * [DQN](examples/keras-rl/fishing.py)
- [tensorflow/agents](/tf-agents)
  * [DQN](https://github.com/boettiger-lab/gym_fishing/blob/master/examples/tf-agents/dqn_fishing-v0.py)
- [stable baselines](/stable-baselines)
  * [PPO](stable-baselines/stable-baselines-ppo.Rmd)

Note that different frameworks have seperate and often conflicting dependencies.  See the `requirements.txt` file in each example, or better, the official doucmentation for each framework.  Consider using virtual environments to avoid conflicts.

## Theory

The optimal dynamic management solution for the stochastic fisheries model is a "constant escapement" policy, as proven by [Reed 1979](https://doi.org/10.1016/0095-0696(79)90014-7).  For small noise, this corresponds to the same 'bang-bang' solution for the determinstic model, proven by [Clark 1973](https://doi.org/10.1086/260090).  Ignoring discounting, the long-term harvest under the constant escapement solution corresponds to the Maximum Sustainable Yield, or MSY, which is the optimal 'constant mortality' solution (i.e. under the constraint of having to harvest a fixed fraction _F_ of the stock each year), as demonstrated independently by [Schaefer 1954](https://doi.org/10.1007/BF02464432) and [Gordon 1954](https://doi.org/10.1086/257497). 

The biomass at MSY can trivially be solved for by maximizing the growth function $X_{t+1} = f(X_t)$.  Discretizing the state space, the dynamic optimal harvest can easily be found by stochastic dynamic programming.  

Here, we seek to compare the performance of modern RL methods, which make no a-priori assumptions about the stock recruitment function, to this known optimal solution (given the underlying population dynamics).  

