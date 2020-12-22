from gym_fishing.envs.base_fishing_env import BaseFishingEnv


class FishingCtsEnv(BaseFishingEnv):
    def __init__(
        self, r=0.3, K=1, sigma=0.0, init_state=0.75, Tmax=100, file=None
    ):
        super().__init__(
            params={"r": r, "K": K, "sigma": sigma, "x0": init_state},
            Tmax=Tmax,
            file=file,
        )
