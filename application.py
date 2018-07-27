from flask import render_template, redirect, url_for
import mc.tcf.config as config
from mc.tcf import manager
from flask import Flask, jsonify, json, request
from mc.tcf import utils
import os
from mc.tcf.ml.models import ColumnSelectTransformer, ResidualEstimator
from sklearn.externals import joblib

application = Flask(__name__)

if config.app_config["force_model_download"] or not os.path.isfile(config.app_config["ml_model_name"]):
    utils.download_file_from_google_drive(config.app_config["ml_base_url"], config.app_config["ml_model_name"])

if config.app_config['ml_load_model']:
    ml_model_path = config.app_config['ml_model_name']
    ensemble = joblib.load(ml_model_path)
    ml = manager.MLManager(config, ensemble)
    sm = manager.SegmentsManager(config)


@application.route('/thecyclingfeast')
def thecyclingfeast():
    return render_template('index.html')

@application.route('/')
def index():
  return redirect(url_for("thecyclingfeast"))

@application.route('/training/<region>/predict', methods=['GET'])
def predict(region=None):
  period = request.args.get('period')
  month_year = list(map(lambda x: int(x), period.split('/')))
  segments = sm.prepare_for_predict(region, month_year)
  y_hat = ml.predict(segments)
  geojson =  ml.do_density(segments, y_hat, region, period)
  jsonStr = json.dumps(geojson)
  return jsonify(jsonStr)

@application.route('/version')
def version():
  return 'The Cycling Feast v. 1.0'

if __name__ == '__main__':
  application.run()
