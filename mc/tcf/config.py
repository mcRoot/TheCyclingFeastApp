import os

messages = {
    "errors": {
    },
    "info": {
    }
}

app_config = {
    "load_model": os.environ.get('LOAD_MODEL', False),
    "ml_trained_model": "ensemble_rf_estimator.pkl",
    "base_dir": "ml"
}