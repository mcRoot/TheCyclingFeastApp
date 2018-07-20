import pandas as pd
import numpy as np

cols = ['Dow', 'Month', 's_lat', 's_lng', 'Num_hotel_30km', "Year_since_kickoff", 'mean_hotel_distances', 'std_hotel_distances',
       'median_hotel_distances', '25_hotel_distances', '75_hotel_distances']
mock_data = pd.DataFrame(np.array([[6, 7, 45, 13, 60, 9, 10, 1, 2, 3, 13],
                          [1, 10, 40, 10, 24, 9, 10, 1, 2, 3, 13],
                          [1, 10, 40, 10, 24, 9, 10, 1, 2, 3, 13],
                          [1, 10, 40, 10, 24, 9, 10, 1, 2, 3, 13],
                          [1, 10, 40, 10, 24, 9, 10, 1, 2, 3, 13],
                          [1, 10, 40, 10, 24, 9, 10, 1, 2, 3, 13],
                          [1, 10, 40, 10, 24, 9, 10, 1, 2, 3, 13],
                          [1, 10, 40, 10, 24, 9, 10, 1, 2, 3, 13],
                          [1, 10, 40, 10, 24, 9, 10, 1, 2, 3, 13],
                          [1, 10, 40, 10, 24, 9, 10, 1, 2, 3, 13],
                          [1, 10, 40, 10, 24, 9, 10, 1, 2, 3, 13]]), columns=cols)
print(mock_data)