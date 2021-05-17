import numpy as np


class BitScaler:
    def __init__(self, bits):
        self._offset: int = 2 ** (bits - 1)
        self._scale: int = 2 ** (-bits)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x + self._offset) * self._scale

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x / self._scale - self._offset
