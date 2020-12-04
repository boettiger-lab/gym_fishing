## Work with this gym in R directly using `reticulate`
library(reticulate)

## Python dependencies
gym         <- import ("gym")
gym_fishing <- import("gym_fishing")
sb3         <- import ("stable_baselines3")

# train an agent (model) on one of the environments:
env <- gym$make("fishing-v1")
model <- sb3$PPO('MlpPolicy', env, verbose=0L) # Must use L for integers!
model$learn(total_timesteps=200000L)
df <- env$simulate(model) # result is a data.frame

