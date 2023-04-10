import numpy as np
import matplotlib.pyplot as plt
from pymdp import utils
from pymdp.maths import softmax



def show_payoff_structure(payoff_structure: tuple[float, float, float]):
    mu1, mu2, sigma = payoff_structure

    x1 = np.linspace(mu1 - 3*sigma, mu1 + 3*sigma, 100)
    x2 = np.linspace(mu2 - 3*sigma, mu2 + 3*sigma, 100)

    if sigma == 0:
        raise NotImplementedError
    else:
        y1 = np.exp(-0.5 * ((x1 - mu1) / sigma)**2) / (sigma * np.sqrt(2*np.pi))
        y2 = np.exp(-0.5 * ((x2 - mu2) / sigma)**2) / (sigma * np.sqrt(2*np.pi))

    # Create the figure and axis objects
    fig, ax = plt.subplots()

    # Plot the first Gaussian distribution
    ax.plot(x1, y1, label='High reward')

    # Plot the second Gaussian distribution
    ax.plot(x2, y2, label='Low reward')

    # Set the title and axis labels
    ax.set_title('Payoff structure')
    ax.set_xlabel('X')
    ax.set_ylabel('Probability Density')

    # Add a legend to the plot
    ax.legend()

    # Show the plot
    plt.show()


def plot_returns(payoffs: np.ndarray):
    mean = np.mean(payoffs, axis=0)
    std_err = np.std(payoffs, axis=0) / np.sqrt(payoffs.shape[0])
    fig, ax = plt.subplots()
    ax.plot(mean)

    ax.fill_between(range(mean.shape[0]), mean + std_err, mean - std_err, alpha=0.8, color='xkcd:light blue')

    plt.ylim(0,1)


class MDPGenerativeModel:

    def __init__(self):
        self.context_names = ['Left-Better', 'Right-Better']
        self.choice_names = ['Start', 'Left Arm', 'Right Arm']

        self.num_states = [len(self.context_names), len(self.choice_names)]
        self.num_factors = len(self.num_states)

        self.context_action_names = ['Do-nothing']
        self.choice_action_names = ['Move-start', 'Play-left', 'Play-right']

        self.num_controls = [len(self.context_action_names), len(self.choice_action_names)]

        self.reward_obs_names = ['Null', 'Loss', 'Reward']
        self.choice_obs_names = ['Start', 'Left Arm', 'Right Arm']

        self.num_obs = [len(self.reward_obs_names), len(self.choice_obs_names)]
        self.num_modalities = len(self.num_obs)
    
    def create_A(self, p_reward=0.8):

        # initialize A
        A = utils.obj_array(self.num_modalities)

        A_reward = np.zeros((len(self.reward_obs_names), len(self.context_names), len(self.choice_names)))

        for choice_id, choice_name in enumerate(self.choice_names):
            if choice_name == 'Start':
                A_reward[0,:,choice_id] = 1.0
            
            elif choice_name == 'Left Arm':
                A_reward[1:,:,choice_id] = np.array([ [1.0-p_reward, p_reward], 
                                                    [p_reward, 1.0-p_reward]])
            elif choice_name == 'Right Arm':
                A_reward[1:, :, choice_id] = np.array([[ p_reward, 1.0- p_reward], 
                                                    [1- p_reward, p_reward]])
        
        A[0] = A_reward

        A_choice = np.zeros((len(self.choice_obs_names), len(self.context_names), len(self.choice_names)))

        for choice_id in range(len(self.choice_names)):
            A_choice[choice_id, :, choice_id] = 1.0

        A[1] = A_choice

        return A


    def create_B(self, p_change=0.0):
        B = utils.initialize_empty_B(self.num_states, self.num_states)

        # Context transitions (uncontrollable) 
        B_context = np.zeros( (len(self.context_names), len(self.context_names), len(self.context_action_names)) )
        B_context[:,:,0] = np.array([[1.-p_change,    p_change], 
                                    [p_change, 1.-p_change]]
                                    )
        
        # Choice transitions (controllable)
        B_choice = np.zeros( (len(self.choice_names), len(self.choice_names), len(self.choice_action_names)) )
        for choice_i in range(len(self.choice_names)):
            B_choice[choice_i, :, choice_i] = 1.0

        B[0], B[1] = B_context, B_choice

        return B


    def create_C(self, reward=2., pun=-4.):
        """ Creates the C array, AKA the observation prior for the MAB task, parameterized by a `reward` and `pun` (punishment) parameter """

        C = utils.obj_array_zeros(self.num_obs)
        C[1] = np.array([0., pun, reward])
        return C


    def create_D(self, p_context=0.5):
        """
        Creates the D array AKA the hidden state prior at the first timestep for the MAB task, parameterized by a `p_context` parameter that
        parameterizes the agent's prior beliefs about whether the context is "Left Arm Better" at the first timestep of a given trial
        """

        D = utils.obj_array(self.num_factors)

        """ Context prior """
        D_context = np.array([p_context, 1. - p_context])
        D[0] = D_context 


        """ Choice-state prior """
        D_choice = np.zeros(len(self.choice_names))
        D_choice[self.choice_names.index("Start")] = 1.0
        D[1] = D_choice

        return D