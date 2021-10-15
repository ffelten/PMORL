import scipy.special
import numpy as np


def softmax(values) -> int:
    probabilities = __softm(values)

    return np.random.choice(np.argwhere(values == np.random.choice(values, 1, p=probabilities)).flatten())

def __softm(x):
    # trick for numerical stability
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
