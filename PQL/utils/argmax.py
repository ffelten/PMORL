import numpy as np


def argmax(values) -> int:
    best_values = np.argwhere(values == np.amax(values)).flatten()

    return np.random.choice(best_values)
