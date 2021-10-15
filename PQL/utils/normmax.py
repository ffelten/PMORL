import numpy as np

def normmax(values) -> int:
    """
    Normalize the values to sum == 1. Returns an index in values following the probability distribution of the normalized values
    """
    values = np.array(values, dtype='float64')

    if np.isinf(values).any():
        values[np.isinf(values)] = np.finfo(np.float32).max

    probabilities = values / values.sum()

    return np.random.choice(np.argwhere(values == np.random.choice(values, 1, p=probabilities)).flatten())

def __norm(x):
    return x / x.sum(axis=0)
