from gym.envs.registration import register

register(
    id="fishing-v0",
    entry_point="gym_fishing.envs:FishingEnv",
)


register(
    id="fishing-v1",
    entry_point="gym_fishing.envs:FishingCtsEnv",
)

register(
    id="fishing-v2",
    entry_point="gym_fishing.envs:FishingTippingEnv",
)

register(
    id="fishing-v4",
    entry_point="gym_fishing.envs:FishingModelError",
)


register(
    id="fishing-v5",
    entry_point="gym_fishing.envs:Allen",
)

register(
    id="fishing-v6",
    entry_point="gym_fishing.envs:BevertonHolt",
)

register(
    id="fishing-v7",
    entry_point="gym_fishing.envs:May",
)

register(
    id="fishing-v8",
    entry_point="gym_fishing.envs:Myers",
)

register(
    id="fishing-v9",
    entry_point="gym_fishing.envs:Ricker",
)

register(
    id="fishing-v10",
    entry_point="gym_fishing.envs:NonStationary",
)

register(
    id="fishing-v11",
    entry_point="gym_fishing.envs:ModelUncertainty",
)
