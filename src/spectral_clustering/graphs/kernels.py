import numpy as np

def gauss_similarity(dists, sigma):
    return np.exp(-dists**2 / (2*sigma**2))