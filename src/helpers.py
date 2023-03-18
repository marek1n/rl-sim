import numpy as np
import matplotlib.pyplot as plt



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
