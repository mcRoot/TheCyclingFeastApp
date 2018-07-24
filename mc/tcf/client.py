import requests
import numpy as np
import os
from sklearn.externals import joblib

class MLClient:

    def __init__(self, config):
        self.config = config
        if self.config.app_config['ml_load_model']:
            self.ensemble = self._load_model()

    def _load_model(self):
        ml_model_path = self.config.app_config['ml_model_name']#os.path.join(self.config.app_config['ml_base_dir'], self.config.app_config['ml_model_name'])
        return joblib.load(ml_model_path)

    def predict(self, X=None):
        return self.ensemble.predict(X)
