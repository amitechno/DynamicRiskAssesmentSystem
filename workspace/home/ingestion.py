import pandas as pd
import os
import json
from datetime import datetime

# Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = os.path.join(os.getcwd(),config['input_folder_path'])
output_folder_path = os.path.join(os.getcwd(),config['output_folder_path'])
prod_deployment_path = os.path.join(os.getcwd(),config['prod_deployment_path']) 
outputfile = 'finaldata.csv'
datarecord = 'ingestedfiles.txt'


# Function for data ingestion
def merge_multiple_dataframe(input_data, outputfile):
    filenames = os.listdir(input_data)
    ingested_file_details = []
    df_list = pd.DataFrame(columns=['corporation','lastmonth_activity','lastyear_activity','number_of_employees','exited'])
    
    for each_filename in filenames:
        df1 = pd.read_csv(input_folder_path + each_filename) 
        df_list=df_list.append(df1).reset_index(drop=True)

        # Get the current date and time
        current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')        
        # Append the file details to the list
        ingested_file_details.append(f"File: {each_filename}, Ingested at: {current_datetime}, Length: {len(df1.index)} records")
        
          
    result=df_list.drop_duplicates()
    result.to_csv(output_folder_path+outputfile, index=False)
    
    # Write the ingested file details to the datarecord text file
    
    with open(prod_deployment_path+datarecord, 'w') as f:
        f.write('\n'.join(ingested_file_details))
        f.write('\n')  # Add an extra line break for separation

 

if __name__ == '__main__':
    merge_multiple_dataframe(input_folder_path, outputfile)
