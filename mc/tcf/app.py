from flask import Flask, render_template, request, redirect, url_for, make_response
import mc.tcf.config as config
from mc.tcf.ml.models import ColumnSelectTransformer, ResidualEstimator
from mc.tcf import manager
from flask import Flask,jsonify,json
from mc.tcf import utils
import os

app = Flask(__name__)

if not os.path.isfile(config.app_config["ml_model_name"]):
    utils.download_file_from_google_drive(config.app_config["ml_base_url"], config.app_config["ml_model_name"])
ml = manager.MLManager(config)
sm = manager.SegmentsManager(config)

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
  from mc.tcf.ml.models import ColumnSelectTransformer, ResidualEstimator
  app.run(port=8080)
