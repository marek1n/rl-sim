import numpy as np
from pymdp import utils


class Env:
    def __init__(self, 
                 n_agents : int = 1, 
                 n_options: int = 2, 
                 payoff_structure: tuple[float, ...] = (0.6, 0.59, 0)
                 ):
        self.n_agents = n_agents # TODO
        self.n_options = n_options
        # payoffs (rewards)
        self.payoff_better, self.payoff_worse, \
            self.payoff_sd = payoff_structure
        
        # best action drawn at random at init TODO: hardcode options?
        self.best_action = np.random.randint(0, n_options)

        # record history of actions performed on the environment
        self.history = [] #dict with id as kz?

    def return_rewards(self, choice_idx: int) -> tuple[float, bool]:
        assert isinstance(choice_idx, int) and (choice_idx < self.n_options)
        # TODO
        better = np.random.normal(self.payoff_better, self.payoff_sd)
        worse = np.random.normal(self.payoff_worse, self.payoff_sd)

        return (better, True) if (choice_idx == self.best_action) else (worse, False)
    
    def switch(self):
         self.best_action = np.random.choice(
             np.setdiff1d(np.arange(self.n_options), self.best_action)
             )
         
    def reset(self):
        self.history = []
    
    def __repr__(self):
        return f"""MAB env\nAgents: {self.n_agents}\nSize of action space: {self.n_options}
Best idx: {self.best_action}
Rewards (high, low, SD): {self.payoff_better, self.payoff_worse, self.payoff_sd}"""



class Env_ActInf:

  def __init__(self, context = None, p_reward = 0.8, p_change = 0.3):

    self.context_names = ["Left-Better", "Right-Better"]

    if context == None:
      self.context = self.context_names[utils.sample(np.array([0.5, 0.5]))] # randomly sample which bandit arm is better (Left or Right)
    else:
      self.context = self.context_names[context]

    self.p_reward = p_reward

    self.reward_obs_names = ['Null', 'Loss', 'Reward']

    self.p_change=p_change # does nothing now

  def step(self, action, change=False):

    # change the context stochastically at each timestep
    # change_or_stay = utils.sample(np.array([self.p_change, 1. - self.p_change]))
    if change:
      if self.context == 'Left-Better':
        self.context = 'Right-Better'
      elif self.context == 'Right-Better':
        self.context = 'Left-Better'

    if action == "Move-start":
      observed_reward = "Null"
      observed_choice = "Start"

    elif action == "Play-left":
      observed_choice = "Left Arm"
      if self.context == "Left-Better":
        observed_reward = self.reward_obs_names[utils.sample(np.array([0.0, 1.0 - self.p_reward, self.p_reward]))]
      elif self.context == "Right-Better":
        observed_reward = self.reward_obs_names[utils.sample(np.array([0.0, self.p_reward, 1.0 - self.p_reward]))]

    elif action == "Play-right":
      observed_choice = "Right Arm"
      if self.context == "Right-Better":
        observed_reward = self.reward_obs_names[utils.sample(np.array([0.0, 1.0 - self.p_reward, self.p_reward]))]
      elif self.context == "Left-Better":
        observed_reward = self.reward_obs_names[utils.sample(np.array([0.0, self.p_reward, 1.0 - self.p_reward]))]
    
    obs = [observed_reward, observed_choice]

    return obs
  