from mc.tcf import client
import os
import pandas as pd
from datetime import datetime
import numpy as np

class MLManager():

    def __init__(self, config):
        self.config = config
        self.ml_client = client.MLClient(config)

    def predict(self, X):
        return self.ml_client.predict(X)

    def do_kde(self, X, y_hat, region):
        X['num_trainings'] = y_hat
        df = X.groupby("segment").aggregate({"num_trainings": "sum", "s_lat": "first", "s_lng": "first"})
        df_kde = df.set_index(['s_lat', 's_lng'])['num_trainings'].repeat(np.log(df['num_trainings']).astype(int)).reset_index()
        df_sorted = df.sort_values("num_trainings", ascending=False)[0:10]
        best_segments = X[X['segment'].isin(df_sorted.index.values)].sort_values(["segment", "Dow"])[['s_lat', 's_lng', "loc_name", "name", "Dow", "segment", "num_trainings"]]
        res = np.vstack([df_kde['s_lat'], df_kde['s_lng']]).T
        return {"res": res.tolist()}


class SegmentsManager():

    def __init__(self, config):
        self.segments = pd.read_csv(os.path.join(config.app_config['segments_base_dir'], config.app_config['segments_csv_name']))
        self.config = config


    def _get_segments_for_region_code(self, region_code=None):
        if region_code is None:
            raise ValueError('No region code was specified')
        return self.segments[self.segments.regional_code == region_code.upper()]

    def prepare_for_predict(self, region_code=None, month=None):
        '''
        Extracts segments belonging to a give region then makes the necessary
        arrangements to prepare data for machine learning
        each segment is replicated for each day of the week (column Dow) and
        column Month is added dependending on the month one wants to predict
        number of rides
        :param region_code: the code of the region to extract segments from
        :param month the month one wants to predict number of rides in
        :return: the dataframe ready to be submitted to ml mode to do predictions
        '''
        if month is None:
            raise ValueError('Month is mandatory to predict rides')
        df = pd.DataFrame()
        ref_segments = self._get_segments_for_region_code(region_code)
        df = pd.concat([ref_segments] * 7, ignore_index=True)
        df['Month'] = month
        dow = pd.Series([0, 1, 2, 3, 4, 5, 6])
        df['Dow'] = dow.append([dow] * ref_segments.shape[0], ignore_index=True)
        df['Year_since_kickoff'] = datetime.now().year - self.config.app_config['strava_kickoff_year']
        return df

