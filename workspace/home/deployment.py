import os
import shutil
import json




##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 
 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
model_path = os.path.join(config['output_model_path']) 
model_file = model_path+'trainedmodel.pkl'


####################function for deployment
def store_model_into_pickle():
    source_files = ['latestscore.txt', 'ingestedfiles.txt', model_file]
    for file_name in source_files:
        source_file_path = os.path.join(os.getcwd(), file_name)  # Construct the source file path
        shutil.copy(source_file_path, prod_deployment_path)

if __name__ == '__main__':
    store_model_into_pickle()
        
        

