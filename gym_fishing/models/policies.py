

class msy:
    def __init__(self, env):
      self.env = env

    def predict(self, obs):
      msy = self.env.r * self.env.K / 4
      action = self.env.get_action(msy)
      return action, obs

class escapement:
    def __init__(self, env):
      self.env = env

    def predict(self, obs):
      fish_population = self.env.get_fish_population(obs)
      quota = max(fish_population - self.env.K / 2., 0.) 
      action = self.env.get_action(quota)
      return action, obs

class user_action:
    def __init__(self, env):
      self.env = env

    def predict(self, obs):
      fish_population = self.env.get_fish_population(obs)
      prompt = "fish population: " + str(fish_population) + ". Your harvest quota: "
      quota = input(prompt)
      action = self.env.get_action(float(quota))
      return action, obs


