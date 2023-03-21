import numpy as np



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
        
        # best action drawn at random at init
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
    
    def __repr__(self):
        return f"""MAB env\nAgents: {self.n_agents}\nSize of action space: {self.n_options}
Best idx: {self.best_action}
Rewards (high, low, SD): {self.payoff_better, self.payoff_worse, self.payoff_sd}"""
