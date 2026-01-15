import numpy as np

class BaseFilter:
    """Abstract base class for modular filter implementation."""
    def __init__(self, x0, P0):
        self.x = x0  # State estimate:
        self.P = P0  # State covariance:

    def predict(self, **kwargs):
        raise NotImplementedError

    def update(self, z, **kwargs):
        raise NotImplementedError