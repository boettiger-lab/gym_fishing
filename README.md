![Install & Train](https://github.com/boettiger-lab/open_ai_fishing/workflows/Install%20&%20Train/badge.svg) [![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)

Here is the fishing gym. Work IN PROGRESS!

## Environments

So far, we have:

Simple fishing model defined in a continuous state space of fish biomass, with:

- Discrete action space with three actions: maintain harvest level, increase harvest by 20%, decrease harvest by 20%
- Discrete action space with n > 3 actions: action is taken as quota, `quota = action / n_actions * K`
- Continuous action space, `action` = quota.

## Examples

Examples for running this gym environment in several frameworks:

- [keras-rl](/keras-rl)
  * [DQN]()
- [tensorflow/agents](/tf-agents)
  * ...
- [stable baselines](/stable-baselines)
  * ...



