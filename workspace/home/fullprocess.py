import os
import json
from ingestion import *
from diagnostics import *
from scoring import *
from training import *
import sys
from sklearn.metrics import accuracy_score
import pickle

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Define paths
source_data_path = os.path.join(config['input_folder_path'])
prod_folder_path = os.path.join(config['prod_deployment_path'])
ingested_files_path = prod_folder_path + 'ingestedfiles.txt'
print(f"ingested file path: ",ingested_files_path ) 
outputfile = 'finaldata.csv'
datafile = os.path.join(config['output_folder_path'], outputfile)

# Read the list of previously ingested files into a set
ingested_files = set()
with open(ingested_files_path, 'r') as file:
    ingested_files.update([line.split(', ')[0].replace('File: ', '') for line in file.read().splitlines()])
print(f"Existing files:", ingested_files)
# Get a list of all files in the source data directory
source_files = [f for f in os.listdir(source_data_path) if os.path.isfile(os.path.join(source_data_path, f))]

# Determine new files to ingest
new_files_to_ingest = [f for f in source_files if f not in ingested_files]

if new_files_to_ingest:
    print(f"New files found, ingesting data from: {new_files_to_ingest}")
    # Data ingestion
    merge_multiple_dataframe(source_data_path, outputfile)
else:
    print("No new files found. Process ends.")
    sys.exit(0)  # Exit the program

##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
with open(os.path.join(prod_folder_path, 'latestscore.txt')) as file:
    last_f1_score_str = file.read().strip()  # Read and remove leading/trailing whitespace
    last_f1_score = float(last_f1_score_str.split(': ')[1])  # Extract and convert the numeric part

# Score the new data
new_f1_score = score_model(datafile)
print(f"last F1_Score was: {last_f1_score}. New F1_score is: {new_f1_score} ")
if new_f1_score < last_f1_score:
    print("Model drift: Deploy the best model")
    train_model()
    # run diagnostics.py and reporting.py for the re-deployed model
    os.system("python diagnostics.py")
    os.system("python reporting.py")
    os.system("python apicall.py")
else:
    print("No model drifts: Process ends.")
    sys.exit(0)  # Exit the program

