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

    def population_draw(self):
        self.fish_population = allen(self.fish_population, self.params)
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
        self.fish_population = beverton_holt(self.fish_population, self.params)
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

    def population_draw(self):
        self.fish_population = myers(self.fish_population, self.params)
        return self.fish_population


class May(BaseFishingEnv):
    def __init__(
        self,
        r=0.8,
        K=1.0,
        q=2,
        b=0.131,
        sigma=0.05,
        a=0.2,
        init_state=0.75,
        Tmax=100,
        file=None,
    ):
        super().__init__(
            params={"r": r, "K": K, "sigma": sigma, "q": q, "b": b, "a": a},
            init_state=init_state,
            Tmax=Tmax,
            file=file,
        )

    def population_draw(self):
        self.fish_population = may(self.fish_population, self.params)
        return self.fish_population


class Ricker(BaseFishingEnv):
    def __init__(self, r=0.3, K=1, sigma=0.1, init_state=0.75, Tmax=100, file=None):
        super().__init__(
            params={"r": r, "K": K, "sigma": sigma},
            init_state=init_state,
            Tmax=Tmax,
            file=file,
        )

    def population_draw(self):
        self.fish_population = ricker(self.fish_population, self.params)
        return self.fish_population


class NonStationary(BaseFishingEnv):
    def __init__(
        self, r=0.8, K=1, sigma=0.1, alpha=-0.007, init_state=0.75, Tmax=100, file=None
    ):
        super().__init__(
            params={"r": r, "K": K, "sigma": sigma, "alpha": alpha},
            init_state=init_state,
            Tmax=Tmax,
            file=file,
        )

    def population_draw(self):
        self.params["r"] = self.params["r"] + self.params["alpha"]
        self.fish_population = beverton_holt(self.fish_population, self.params)
        return self.fish_population


class ModelUncertainty(BaseFishingEnv):
    def __init__(
        self,
        r=0.8,
        K=1.0,
        theta=3,
        C=0.5,
        q=2,
        b=0.131,
        sigma=0.05,
        a=0.2,
        init_state=0.75,
        Tmax=100,
        file=None,
    ):
        super().__init__(
            params={
                "r": r,
                "K": K,
                "sigma": sigma,
                "q": q,
                "b": b,
                "a": a,
                "C": C,
                "theta": theta,
            },
            init_state=init_state,
            Tmax=Tmax,
            file=file,
        )
        self.model = np.random.choice(
            ["allen", "beverton_holt", "myers", "may", "ricker"]
        )

    def population_draw(self):
        f = population_model[self.model]
        self.fish_population = f(self.fish_population, self.params)
        return self.fish_population

    def reset(self):
        self.state = np.array([self.init_state / self.K - 1])
        self.fish_population = self.init_state
        self.model = np.random.choice(
            ["allen", "beverton_holt", "myers", "may", "ricker"]
        )
        self.years_passed = 0
        self.reward = 0
        self.harvest = 0
        return self.state


## Growth Functions ##
def allen(x, params):
    with np.errstate(divide="ignore"):
        mu = (
            np.log(x)
            + params["r"] * (1 - x / params["K"]) * (1 - params["C"]) / params["K"]
        )
    return np.maximum(0, np.random.lognormal(mu, params["sigma"]))


def beverton_holt(x, params):
    A = params["r"] + 1
    B = params["K"] / params["r"]
    with np.errstate(divide="ignore"):
        mu = np.log(A) + np.log(x) - np.log(1 + x / B)
    return np.maximum(0, np.random.lognormal(mu, params["sigma"]))


def myers(x, params):
    ## FIXME this rescaling does not hold for theta != 1
    A = params["r"] + 1
    B = params["K"] / params["r"]
    with np.errstate(divide="ignore"):
        mu = (
            np.log(A)
            + params["theta"] * np.log(x)
            - np.log(1 + np.power(x, params["theta"]) / np.power(B, params["theta"]))
        )
    return np.maximum(0, np.random.lognormal(mu, params["sigma"]))


def may(x, params):
    with np.errstate(divide="ignore"):
        r = params["r"]
        K = params["K"]
        a = params["a"]
        q = params["q"]
        b = params["b"]
        exp_mu = x * r * (1 - x / K) - a * np.power(x, q) / (
            np.power(x, q) + np.power(b, q)
        )
        exp_mu = np.maximum(0, exp_mu)
        mu = np.log(exp_mu)
    return np.maximum(0, np.random.lognormal(mu, params["sigma"]))


def ricker(x, params):
    with np.errstate(divide="ignore"):
        mu = np.log(x) + params["r"] * (1 - x / params["K"])
    return np.maximum(0, np.random.lognormal(mu, params["sigma"]))


population_model = {
    "allen": allen,
    "beverton_holt": beverton_holt,
    "myers": myers,
    "may": may,
    "ricker": ricker,
}
