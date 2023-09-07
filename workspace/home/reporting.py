import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'])
deployed_model_path = os.path.join(config['prod_deployment_path'])
test_data_path = os.path.join(config['test_data_path'],'testdata.csv')


##############Function for reporting
def analyse_model(test_data_path):
    try:
        # Load the trained model
        with open(deployed_model_path+'trainedmodel.pkl', 'rb') as model_file:
            model = pickle.load(model_file)

        # Load the test data
        test_data = pd.read_csv(test_data_path )

        # Separate features (X) and target (y)
        X_test = test_data.iloc[:, 1:-1].values
        y_test = test_data['exited']

        # Make predictions using the model
        y_pred = model.predict(X_test)

        # Calculate the confusion matrix
        confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

        # Plot the confusion matrix
        plt.figure(figsize=(6, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')

        # Save the confusion matrix plot to a file
        output_path = dataset_csv_path+'confusionmatrix.png'
        plt.savefig(output_path, bbox_inches='tight')

        # Close the plot
        plt.close()

    except Exception as e:
        print("Error:", str(e))

if __name__ == '__main__':
    analyse_model(test_data_path)
