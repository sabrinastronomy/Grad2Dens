import matplotlib.pyplot as plt
import numpy as np

line_search_beta = np.linspace(0.1, 2, 100)

if True:
    posteriors = np.load("posteriors.npy")
    plt.semilogy(line_search_beta, posteriors)
    plt.xlabel("beta")
    plt.ylabel("posteriors")
    plt.savefig("prelim_grid_search.png")
    plt.close()
