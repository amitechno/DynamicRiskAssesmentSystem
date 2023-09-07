from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os
from diagnostics import *
from scoring import *



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 


#######################Prediction Endpoint
@app.route("/prediction", methods=["POST"])
def predict():        
    input_data = request.args.get('dataset_location')
    predicted = model_predictions(input_data)
    return jsonify({"predictions": predicted.tolist()})

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scores():    
    input_data = request.args.get('dataset_location')
    score = score_model(input_data)
    return jsonify({"f1_score": score})

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    stats = dataframe_summary()
    return jsonify(stats)

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():        
    missing_values = missing_data_percentage()
    time_taken = execution_time()
    outdated_package = outdated_packages_list()
    return str(f"execution_time: {time_taken} \n missing_values: {missing_values} \n outdated_packages: {outdated_package}")

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
