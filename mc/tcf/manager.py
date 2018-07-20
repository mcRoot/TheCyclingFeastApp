from mc.tcf import client
from mc.tcf import config

class Manager():

    def __init__(self):
        self.ml_client = client.MLClient(config)

    def predict(self, X):
        return self.ml_client.predict(X)