from flask import render_template, redirect, url_for
import mc.tcf.config as config
from mc.tcf import manager
from flask import Flask,jsonify,json
from mc.tcf import utils
import os
from mc.tcf.ml.models import ColumnSelectTransformer, ResidualEstimator
from sklearn.externals import joblib


if config.app_config["force_model_download"] or not os.path.isfile(config.app_config["ml_model_name"]):
    utils.download_file_from_google_drive(config.app_config["ml_base_url"], config.app_config["ml_model_name"])

if config.app_config['ml_load_model']:
    ml_model_path = config.app_config['ml_model_name']
    ensemble = joblib.load(ml_model_path)

ml = manager.MLManager(config, ensemble)
sm = manager.SegmentsManager(config)

application = Flask(__name__)
app = application

@app.route('/thecyclingfeast')
def thecyclingfeast():
    return render_template('index.html')

@app.route('/')
def index():
  return redirect(url_for("thecyclingfeast"))

@app.route('/training/<region>/predict', methods=['GET'])
def predict(region=None):
  segments = sm.prepare_for_predict(region, 7)
  y_hat = ml.predict(segments)
  geojson =  ml.do_kde(segments, y_hat, region)
  jsonStr = json.dumps(geojson)
  return jsonify(jsonStr)

@app.route('/version')
def version():
  return 'The Cycling Feast v. 1.0'

if __name__ == '__main__':
  app.run(port=8080)
