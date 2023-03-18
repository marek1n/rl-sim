import numpy as np



class AgentBase:
    def __init__(self, id, n_options=2, q_init=0.1):
        # properties
        self.id = id

        # arrays to hold behavioural history
        self.Q_vals = [np.array([q_init for i in range(n_options)])]
        self.choices = []
        self.correct = []
        self.payoffs = []
        
        self.n_actions = n_options # TODO hmm

    def update_state(self, reward, correct):
        self.payoffs.append(reward)
        self.correct.append(correct)

    def __repr__(self):
        return f"Agent {self.id}"
    

class AgentEWA(AgentBase):
    def __init__(self, id, phi=0.1, lam=2):
        super().__init__(id)
        self.phi = phi
        self.lam = lam
        self.choice_f = lambda x: np.exp(x)/sum(np.exp(x)) # TODO parameter?

    def choose_action(self) -> int:
        choice_probabilities = self.choice_f(self.Q_vals[-1])
        choice = np.random.choice(self.n_actions, 1, p=choice_probabilities).item()
        self.choices.append(choice)
        
        return choice

    def update_Qvals(self, choice, reward):
        last_q = self.Q_vals[-1].copy() # get most recent Qvals
        #update all choices?
        for idx, val in enumerate(last_q):
            if idx == choice:
                last_q[idx] = (1 - self.phi) * last_q[choice] + self.phi * reward 
            else:
                #NOTE this makes the agent perform **really** crap
                # last_q[idx] = (1 - self.phi) * last_q[choice] + self.phi * 0
                pass

        self.Q_vals.append(last_q)
    

class AgentQ(AgentBase):
    def __init__(self, id, epsilon=0.5):
        super().__init__(id)
        self.epsilon = epsilon

    def choose_action(self) -> int:
        q_vals_current = self.Q_vals[-1]
        p = np.random.uniform()
        # choose random with p=epsilon otherwise greedy
        if p < self.epsilon:
            choice = np.random.randint(0, self.n_actions)
        else:
            choice = np.argmax(q_vals_current)

        self.choices.append(choice)
        
        return int(choice)
    
    def update_Qvals(self, choice, reward):
        last_q = self.Q_vals[-1].copy() # get most recent Qvals
        chosen_q = last_q[choice]
        new_q = chosen_q + (reward - chosen_q) / len(self.Q_vals) # == n of steps, >= 1
        last_q[choice] = new_q
        
        self.Q_vals.append(last_q)
