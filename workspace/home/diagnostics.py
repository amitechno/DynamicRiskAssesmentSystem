
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess
import time

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

prod_deployment_path = os.path.join(config['prod_deployment_path'])

dataset_csv_path = os.path.join(config['output_folder_path'],'finaldata.csv')



##################Function to get model predictions
def model_predictions(dataset_path):
    dataset = pd.read_csv(dataset_path)
    print(dataset)
    with open(prod_deployment_path + 'trainedmodel.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    X = dataset.iloc[:, 1:-1].values
    # Make predictions using the model
    predicted = model.predict(X)
    return predicted

##################Function to get summary statistics
def dataframe_summary():
    dataset = pd.read_csv(dataset_csv_path)
    numeric_columns = dataset.select_dtypes(include=['number'])  # Select numeric columns
    # Calculate summary statistics for numeric columns
    summary_statistics = numeric_columns.describe().transpose()  # Calculate summary statistics

    # Extract means, medians, and standard deviations
    means = summary_statistics['mean'].tolist()
    medians = summary_statistics['50%'].tolist()
    std_devs = summary_statistics['std'].tolist()

    # Create a list containing summary statistics for each numeric column
    summary_list = []
    for col in numeric_columns.columns:
        summary_list.append({
            'Column': col,
            'Mean': means.pop(0),
            'Median': medians.pop(0),
            'Std Dev': std_devs.pop(0)
        })
        
    print(summary_list)
    return summary_list
##################Function to get missing data
def missing_data_percentage():
    dataset = pd.read_csv(dataset_csv_path)
    numeric_columns = dataset.select_dtypes(include=['number'])  # Select numeric columns
    for columns in numeric_columns:
        nas = list(dataset.isna().sum())
        napercents = [nas[i] / len(dataset.index)
                      for i in range(len(nas))]
    print( napercents )

##################Function to get timings
def execution_time():
    # Measure the execution time of data ingestion (ingestion.py)
    start_time_ingestion = time.time()
    subprocess.run(["python", "ingestion.py"]) 
    end_time_ingestion = time.time()
    ingestion_time = end_time_ingestion - start_time_ingestion

    # Measure the execution time of model training (training.py)
    start_time_training = time.time()
    subprocess.run(["python", "training.py"]) 
    end_time_training = time.time()
    training_time = end_time_training - start_time_training
    print('ingestion_time:\t', ingestion_time)
    print('training_time:\t', training_time)
    return [ingestion_time, training_time]

##################Function to check dependencies
def outdated_packages_list():
    try:
        # Get the list of installed packages and their versions
        installed_packages = subprocess.check_output(["python3", "-m", "pip", "freeze"]).decode("utf-8").split("\n")

        # Create a dictionary to store the current versions of installed packages
        current_versions = {}
        for package_info in installed_packages:
            package_info = package_info.strip()
            if package_info:
                package_name, package_version = package_info.split("==")
                current_versions[package_name] = package_version

        # Read the requirements.txt file and parse package names and required versions
        with open('requirements.txt', 'r') as f:
            requirements = f.read().split("\n")

        # Create a dictionary to store the required versions from requirements.txt
        required_versions = {}
        for requirement in requirements:
            requirement = requirement.strip()
            if requirement:
                package_info = requirement.split("==")
                if len(package_info) == 2:
                    package_name, package_version = package_info
                    required_versions[package_name] = package_version

        # Compare current versions with required versions
        dependency_table = []
        for package_name, required_version in required_versions.items():
            current_version = current_versions.get(package_name, "Not Installed")

            # Get the latest version using 'pip show' command
            latest_version = subprocess.check_output(["python3", "-m","pip", "show", package_name]).decode("utf-8")
            latest_version = next((line.split(': ')[1] for line in latest_version.split('\n') if line.startswith('Version: ')), "Not Found")

            dependency_table.append([package_name, current_version, required_version, latest_version])

        # Print the dependency table
        for package_info in dependency_table:
            print("{:<30} {:<15} {:<15} {:<15}".format(package_info[0], package_info[1], package_info[2], package_info[3]))

    except Exception as e:
        print(str(e))
    
    



if __name__ == '__main__':
    test = model_predictions(dataset_csv_path)
    print(test)
    dataframe_summary()
    missing_data_percentage()
    execution_time()
    outdated_packages_list()





    
