import pickle

import numpy as np


class SKLearnScorer:
    """
    Scorer class for Bayesian optimization, based on scikit-learn models
    """

    def __init__(self, path, penalize=False):
        """
        Args:
            path: path to the saved model
            penalize: if True, penalize for values outside of bounds
        """
        with open(path, "rb") as file:
            self.model = pickle.load(file)
        self.penalize = penalize

    def __call__(self, **args) -> float:
        input_vector = list({**args}.values())
        input_vector = np.array(input_vector).reshape(1, -1)
        output = self.model.predict_proba(input_vector)[0][0]
        if self.penalize:
            output = output * gaussian_reward(input_vector, mu=6.47, sigma=2.44)
        return output


def gaussian_reward(vec: np.array, mu: float, sigma: float):
    x = np.linalg.norm(vec)
    c = np.sqrt(2 * np.pi)
    score = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / sigma / c
    return score
