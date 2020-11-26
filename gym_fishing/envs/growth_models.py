import math
import numpy as np
from gym_fishing.envs.base_fishing_env import BaseFishingEnv


class Allen(BaseFishingEnv):
    def __init__(
        self, r=0.3, K=1, C=0.5, sigma=0.1, init_state=0.75, Tmax=100, file=None
    ):
        super().__init__(
            params={"r": r, "K": K, "sigma": sigma, "C": C},
            init_state=init_state,
            Tmax=Tmax,
            file=file,
        )
        self.C = C

    def population_draw(self):
        x = self.fish_population
        with np.errstate(divide='ignore'):
            mu = np.log(x) + self.r * (1 - x / self.K) * (1 - self.C) / self.K
        self.fish_population = np.maximum(0, np.random.lognormal(mu, self.sigma))
        return self.fish_population


class BevertonHolt(BaseFishingEnv):
    def __init__(self, r=0.3, K=1, sigma=0.1, init_state=0.75, Tmax=100, file=None):
        super().__init__(
            params={"r": r, "K": K, "sigma": sigma},
            init_state=init_state,
            Tmax=Tmax,
            file=file,
        )

    def population_draw(self):
        x = self.fish_population
        with np.errstate(divide='ignore'):
            ## This is A,B notation. Re-write in terms of real r and K
            mu = np.log(x + self.r * x / (1 + x / self.K))
        self.fish_population = np.maximum(0, np.random.lognormal(mu, self.sigma))
        return self.fish_population


class Myers(BaseFishingEnv):
    def __init__(
        self, r=0.3, K=1, theta=3, sigma=0.1, init_state=0.75, Tmax=100, file=None
    ):
        super().__init__(
            params={"r": r, "K": K, "sigma": sigma, "theta": theta},
            init_state=init_state,
            Tmax=Tmax,
            file=file,
        )
        self.theta = theta

    def population_draw(self):
        x = self.fish_population
        with np.errstate(divide='ignore'):
            mu = np.log(
                x + self.r * x ^ self.theta / (1 + np.power(abs(x), self.theta) / self.K)
            )
        self.fish_population = np.maximum(0, np.random.lognormal(mu, self.sigma))
        return self.fish_population


class May(BaseFishingEnv):
    def __init__(
        self,
        r=0.8,
        K=153,
        q=2,
        b=20,
        sigma=0.05,
        a=28,
        init_state=0.75,
        Tmax=100,
        file=None,
    ):
        super().__init__(
            params={"r": r, "K": K, "sigma": sigma, "q": 2, "b": 20, "a": 28},
            init_state=init_state,
            Tmax=Tmax,
            file=file,
        )
        self.C = C

    def population_draw(self):
        x = self.fish_population
        with np.errstate(divide='ignore'):
          mu = np.log(
              x * self.r * (1 - x / self.K) - self.a * x
              ^ self.q / (x ^ self.q + self.b ^ self.q)
          )
        self.fish_population = np.maximum(0, np.random.lognormal(mu, self.sigma))
        return self.fish_population


class Ricker(BaseFishingEnv):
    def __init__(
        self, r=0.3, K=1, C=0.5, sigma=0.1, init_state=0.75, Tmax=100, file=None
    ):
        super().__init__(
            params={"r": r, "K": K, "sigma": sigma},
            init_state=init_state,
            Tmax=Tmax,
            file=file,
        )

    def population_draw(self):
        x = self.fish_population
        with np.errstate(divide='ignore'):
            mu = np.log(x) + self.r * (1 - x / self.K)
        self.fish_population = np.maximum(0, np.random.lognormal(mu, self.sigma))
        return self.fish_population
