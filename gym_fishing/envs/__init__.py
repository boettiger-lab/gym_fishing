
from gym.envs.registration import register
from gym_fishing.envs.fishing_cts_env import FishingCtsEnv

register(
    id="fishing-v1",
    entry_point="gym_fishing.envs:FishingCtsEnv",
)

register(
    id="fishing-v2",
    entry_point="gym_fishing.envs.:FishingTippingEnv",
)

register(
    id="fishing-v4",
    entry_point="gym_fishing.envs.fishing_model_error:FishingModelError",
)


register(
    id="fishing-v5",
    entry_point="gym_fishing.envs.growth_models:Allen",
)

register(
    id="fishing-v6",
    entry_point="gym_fishing.envs.growth_models:BevertonHolt",
)

register(
    id="fishing-v7",
    entry_point="gym_fishing.envs.growth_models:May",
)

register(
    id="fishing-v8",
    entry_point="gym_fishing.envs.growth_models:Myers",
)

register(
    id="fishing-v9",
    entry_point="gym_fishing.envs.growth_models:Ricker",
)

register(
    id="fishing-v10",
    entry_point="gym_fishing.envs.growth_models:NonStationary",
)

register(
    id="fishing-v11",
    entry_point="gym_fishing.envs.growth_models:ModelUncertainty",
)
