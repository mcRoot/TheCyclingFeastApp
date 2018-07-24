import unittest
import mc.tcf.manager as managers
import mc.tests.test_config as test_config
import time

from mc.tcf.ml.models import ColumnSelectTransformer, ResidualEstimator

class MLManagerTestCase(unittest.TestCase):

    def setUp(self):
        self.segmentManager = managers.SegmentsManager(test_config)
        self.mlManager = managers.MLManager(test_config)

    def test_predict(self):
        start_time = time.time()
        segments = self.segmentManager.prepare_for_predict('fvg', 7)
        y_hat = self.mlManager.predict(segments)
        end_time = time.time()
        print("Total prediction time {:.3f}s".format(float(end_time - start_time)))
        self.assertTrue(len(y_hat) > 0)
        df = self.mlManager.do_kde(segments, y_hat, 'fvg')
        #self.assertTrue(self.segmentManager._get_segments_for_region_code('fvg').shape[0]< df.shape[0])
        print("Total kde time {:.3f}s".format(float(time.time() - end_time)))
        print(df)


class ManagersTestCase(unittest.TestCase):

    def setUp(self):
        self.segmentManager = managers.SegmentsManager(test_config)

    def test_get_segments_ok(self):
        segments = self.segmentManager._get_segments_for_region_code('sic')
        self.assertTrue(segments.shape[0] > 0)

    def test_get_segments_empty(self):
        segments = self.segmentManager._get_segments_for_region_code('IDN')
        self.assertTrue(segments.shape[0] == 0)

    def test_none_region(self):
        with self.assertRaises(Exception) as context:
            self.segmentManager._get_segments_for_region_code(None)
        self.assertEqual('No region code was specified', context.exception.args[0])

    def test_prepare_predictions(self):
        sum_dow = sum([0, 1, 2, 3, 4, 5, 6])
        segments = self.segmentManager.prepare_for_predict('sic', 7)
        segment_id_1 = segments.iloc[0]['segment']
        self.assertEqual(sum_dow, segments[segments.segment == segment_id_1]['Dow'].sum())
        self.assertEqual(7, segments[segments.segment == segment_id_1].shape[0])
        segment_id_10 = segments.iloc[10]['segment']
        self.assertEqual(sum_dow, segments[segments.segment == segment_id_10]['Dow'].sum())
        self.assertEqual(7, segments[segments.segment == segment_id_10].shape[0])

if __name__ == '__main__':
    unittest.main()