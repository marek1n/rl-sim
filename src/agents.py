import numpy as np
from typing import Any
from collections import Counter



class AgentBase:
    def __init__(self, 
                 id: Any, # needs hashable type
                 n_options: int = 2, 
                 q_init: float = 0.1
                 ):
        # properties
        self.id = id
        self.n_actions = n_options # TODO hmm

        # arrays to hold behavioural history
        self.Q_vals = [np.array([q_init for i in range(n_options)])]
        self.choices = []
        self.correct = []
        self.payoffs = []

    def update_state(self, 
                     reward: tuple[float, ...], 
                     correct: bool
                     ):
        self.payoffs.append(reward)
        self.correct.append(correct)

    def __repr__(self): return f"Agent {self.id}"
    

class AgentEWA(AgentBase):
    def __init__(self, id, phi=0.1, lam=2, sigma=0.5, theta=2):
        super().__init__(id)
        self.phi = phi
        self.lam = lam
        self.theta = theta
        self.ind_choice_f = lambda x: np.exp(x)/sum(np.exp(x))
        self.social_choice_f = lambda p_ind, p_soc: (1 - sigma)*p_ind + sigma*p_soc

    def choose_action(self, obs=None) -> int:
        ind_choice_probabilities = np.array(self.ind_choice_f(self.Q_vals[-1]))

        if obs:
            # assign social probabilities
            counts_dict = Counter(obs.values())
            
            counts = np.array([counts_dict[i] if i in counts_dict.keys() 
                                        else 0 for i in range(self.n_actions)])
            
            soc_choice_probabilities = np.array([(counts[option]**self.theta / sum(counts**self.theta)) 
                                                    for option in range(self.n_actions)])
            
            combined_choice_probabilities = self.social_choice_f(ind_choice_probabilities, soc_choice_probabilities)

            # make choice combining ind and soc
            choice = np.random.choice(self.n_actions, 1, p=combined_choice_probabilities).item()
            
        else:
            choice = np.random.choice(self.n_actions, 1, p=ind_choice_probabilities).item()
        
        
        self.choices.append(choice)
        
        return choice

    def update_Qvals(self, choice, reward):
        last_q = self.Q_vals[-1].copy() # get most recent Qvals
        # individual
        for idx, val in enumerate(last_q):
            if idx == choice:
                last_q[idx] = (1 - self.phi) * last_q[choice] + self.phi * reward 
            else:
                # NOTE can be commented out to change Q-value updating
                last_q[idx] = (1 - self.phi) * last_q[choice] + self.phi * 0
                # pass

        self.Q_vals.append(last_q)
    

class AgentQ(AgentBase):
    def __init__(self, id, epsilon=0.5):
        super().__init__(id)
        self.epsilon = epsilon # TODO anneal?

    def choose_action(self, **kwargs) -> int:
        q_vals_current = self.Q_vals[-1]
        p = np.random.uniform()
        # choose random with p=epsilon otherwise greedy
        if p < self.epsilon:
            choice = np.random.randint(0, self.n_actions)
        else:
            # TODO add social context
            choice = np.argmax(q_vals_current)

        choice = int(choice) # convert from np.int
        self.choices.append(choice)
        
        return choice
    
    def update_Qvals(self, choice, reward):
        last_q = self.Q_vals[-1].copy() # get most recent Qvals
        chosen_q = last_q[choice]
        new_q = chosen_q + (reward - chosen_q) / len(self.Q_vals) # == n of steps, >= 1
        last_q[choice] = new_q
        
        self.Q_vals.append(last_q)
