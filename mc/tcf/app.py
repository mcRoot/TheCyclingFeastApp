from flask import Flask, render_template, request, redirect, url_for, make_response
import mc.tcf.config as config
from mc.tcf.ml.models import ColumnSelectTransformer, ResidualEstimator
from mc.tcf import manager

app = Flask(__name__)
#tm = TableManager(table=app_config["table"], apikey=app_config["apy_key"])
#pm = PlottingManager()

ml = manager.Manager()

@app.route('/thecyclingfeast')
def thecyclingfeast():
    return render_template('index.html')

@app.route('/')
def index():
  return redirect(url_for("thecyclingfeast"))

@app.route('/version')
def version():
  return 'The Cycling Feast v. 1.0'

if __name__ == '__main__':
  app.run(port=8080)
