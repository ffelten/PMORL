import numpy as np

def normmax(values) -> int:
    probabilities = __norm(values)

    return np.random.choice(np.argwhere(values == np.random.choice(values, 1, p=probabilities)).flatten())

def __norm(x):
    return x / x.sum(axis=0)
