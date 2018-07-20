import requests
import numpy as np
import os
from sklearn.externals import joblib

class MLClient:

    def __init__(self, config):
        self.config = config
        if self.config.app_config['load_model']:
            self.ensemble = self._load_model()

    def _load_model(self):
        ml_model_path = os.path.join(self.config.app_config['base_dir'], self.config.app_config['ml_trained_model'])
        return joblib.load(ml_model_path)

    def predict(self, X=None):
        return self.ensemble.predict(X)
