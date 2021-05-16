import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


class DenoisingAutoEncoder(Model):
    def __init__(self, *args, **kwargs) -> Model:
        super().__init__(*args, **kwargs)

        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(320,)),
            layers.Dense(units=240, activation='relu'),
            layers.Dense(units=160, activation='relu'),
        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(units=240, activation='relu'),
            layers.Dense(units=320, activation='sigmoid'),
        ])

    def call(self, x: np.ndarray):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
