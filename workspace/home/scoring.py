from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path'],'testdata.csv') 
model_path = os.path.join(config['output_model_path'])
prod_data_path = os.path.join(config['output_folder_path'],'finaldata.csv') 


#################Function for model scoring
def score_model(data_path):
    # Load the trained model
    with open(model_path + 'trainedmodel.pkl', 'rb') as file:
        model = pickle.load(file)

    # Load test data
    testdata = pd.read_csv(data_path)
    X = testdata.iloc[:, 1:-1].values  # excludes the last column
    y = testdata['exited']

    # Make predictions using the model
    predicted = model.predict(X)

    # Calculate F1 score
    f1score = metrics.f1_score(y, predicted)
    with open('latestscore.txt', 'w') as f:
        f.write(f"F1 Score: {f1score}")
    return f1score

if __name__ == '__main__':
  x =  score_model(prod_data_path)
  print(x)
