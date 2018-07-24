import unittest
from mc.tcf.client import MLClient
import mc.tests.test_config as test_config
import mc.tests.test_data as data

c = 0

def config_client():
    global c
    c = MLClient(test_config)
    print("Client loaded")

class MLClientTestCase(unittest.TestCase):

    def test_predict(self):
        y_pred = c.predict(data.mock_data)
        self.assertEqual(data.mock_data.shape[0], len(y_pred))

if __name__ == '__main__':
    config_client()
    unittest.main()