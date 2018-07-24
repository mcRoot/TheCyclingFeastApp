import requests
import numpy as np
import os

class MLClient:

    def __init__(self, config, ml_model):
        self.config = config
        self.ensemble = ml_model

    def predict(self, X=None):
        return self.ensemble.predict(X)
