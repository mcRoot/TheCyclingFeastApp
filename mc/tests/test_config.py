import os

messages = {
    "errors": {
    },
    "info": {
    }
}

app_config = {
    "ml_load_model": os.environ.get('LOAD_MODEL', True),
    "ml_model_name": "ensemble_rf_estimator.pkl",
    "ml_base_dir": "../tcf/ml/resources",
    "segments_csv_name": "strava_segments.csv",
    "segments_base_dir": "../tcf/resources",
    "strava_kickoff_year": 2009
}

regions = {
    "PIE": [46.464435, 44.060090, 9.214264, 6.626630],
    "VDA": [45.987760, 45.466945, 7.939526, 6.801350],
    "LOM": [46.635185, 44.679649, 11.427699, 8.497861],
    "TAA": [47.091794, 45.673064, 12.477586, 10.381789],
    "LIG": [44.676426, 43.759672, 10.071032, 7.494810],
    "VEN": [46.680475, 44.791218, 13.100977, 10.622963],
    "FVG": [46.647809, 45.580928, 13.918852, 12.320941],
    "EMR": [45.139119, 43.731891, 12.755640, 9.197933],
    "TOS": [44.472690, 42.237669, 12.371355, 9.686721],
    "MAR": [43.994663, 42.682305, 13.926471, 12.369275],
    "UMB": [43.617346, 42.364451, 13.264169, 11.891893],
    "ABR": [42.895082, 41.682106, 14.783012, 13.018486],
    "LAZ": [42.838721, 40.784738, 14.027642, 11.449384],
    "CAM": [41.507373, 39.990560, 15.806445, 13.762112],
    "PUG": [42.226557, 39.789641, 18.520383, 14.934095],
    "BAS": [41.139923, 39.894802, 16.867172, 15.334987],
    "CAL": [40.143929, 37.915756, 17.206527, 15.629687],
    "SIC": [38.812185, 35.491549, 15.652795, 11.925333],
    "SAR": [41.259197, 38.864049, 9.827038, 8.130805],
    "MOL": [42.069830, 41.363964, 15.161560, 13.941014],
}