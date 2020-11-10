[![Install & Train](https://github.com/boettiger-lab/gym_fishing/workflows/Install%20&%20Train/badge.svg)](https://github.com/boettiger-lab/gym_fishing/runs/1323937050?check_suite_focus=true)
[![PyPI Version](https://img.shields.io/pypi/v/gym_fishing)](https://pypi.org/project/gym-fishing/)
![PyPI - Downloads](https://img.shields.io/pypi/dm/gym-fishing)
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)



This repository provides [OpenAI-gym](https://github.com/openai/gym/) class definitions for the classic fisheries management problem.  See [Creating your own Gym Environments](https://github.com/openai/gym/blob/master/docs/creating-environments.md) for details on how a new gym environment is defined. 


## Installation

Install the latest release from pypi:


```
pip install gym_fishing
```


Or install the current dev version by cloning this repo and running:

```
python setup.py sdist bdist_wheel
pip install -e .
```

## Environments


So far, we have: 

A simple fishing model with logistic recruitment, defined in a continuous state space of fish biomass, with:

- `fishing-v0`: A discrete action space with three actions: maintain harvest level, increase harvest by 20%, decrease harvest by 20%. 
- `fishing-v1` A continuous action space, `action` = quota.
- `fishing-v2`: A growth model that contains a tipping point. Uses continuous action space
- `fishing-v3`: Includes observation error. Continouous action space, logistic growth mode.
- `fishing-v4`: Includes parameter uncertainty. 


`fishing-v0` can be configured to allow a discrete action space n > 3 actions, where the action is then treated as a fishing quota, `quota = action / n_actions * K`.  (most frameworks do not support this however.)


## Examples

Examples for running this gym environment in several frameworks:  

- [stable baselines3](/stable-baselines3)    
- [tensorflow/agents](/tf-agents)
- [stable baselines](/stable-baselines) (deprecated)
- [keras-rl](/keras-rl)  (deprecated)


**NOTES**: 

- The [keras-rl](https://github.com/keras-rl/keras-rl) repository is no longer actively maintained.  It still depends on `tensorflow 1.14`, which is deprecated in favor of 1.15.  It's a nice implementation that is easy to learn, but not ideal for production.  Note that all tensorflow 1.x series are supported only on Python 3.7 and earlier.  
- [stable-baselines](https://github.com/hill-a/stable-baselines) is a very popular fork of OpenAI's `baselines`.  Excellent documentation. Uses tensorflow 1.x, and is still actively maintained, though future development effort seems focused on [stable-baselines3](https://github.com/DLR-RM/stable-baselines3), a pytorch-based implementation.
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) is our prefered framework currently.  Well documented and actively developed. Notably, `pytorch` also supports shared GPU usage (eager execution), while tensorflow 1.x default execution reserves the whole GPU.
- [tf-agents](https://github.com/tensorflow/tf-agents) Developed by Google's Tensorflow team (but not an "official" product), supports TF 2.x and 1.x.  Looks like a promising and powerful library, but documentation and ease of use lags behind stable-baselines.  

- [spinning-up](https://github.com/openai/spinningup) is an education-focused implementation from OpenAI that might not be ideal for production use.  

Note that different frameworks have seperate and often conflicting dependencies.  See the `requirements.txt` file in each example, or better, the official doucmentation for each framework.  Consider using virtual environments to avoid conflicts.

## Theory

The optimal dynamic management solution for the stochastic fisheries model is a "constant escapement" policy, as proven by [Reed 1979](https://doi.org/10.1016/0095-0696(79)90014-7).  For small noise, this corresponds to the same 'bang-bang' solution for the determinstic model, proven by [Clark 1973](https://doi.org/10.1086/260090).  Ignoring discounting, the long-term harvest under the constant escapement solution corresponds to the Maximum Sustainable Yield, or MSY, which is the optimal 'constant mortality' solution (i.e. under the constraint of having to harvest a fixed fraction _F_ of the stock each year), as demonstrated independently by [Schaefer 1954](https://doi.org/10.1007/BF02464432) and [Gordon 1954](https://doi.org/10.1086/257497). 

For a given $f(x)$, The biomass at MSY can trivially be solved for by maximizing the growth function $X_{t+1} = f(X_t)$.  Discretizing the state space, the dynamic optimal harvest can easily be found by stochastic dynamic programming.

Here, we seek to compare the performance of modern RL methods, which make no a-priori assumptions about the stock recruitment function, to this known optimal solution (given the underlying population dynamics).  



