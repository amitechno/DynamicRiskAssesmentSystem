import requests

# Specify the base URL of your Flask API
BASE_URL = "http://127.0.0.1:8000"

# API endpoints
PREDICTION_ENDPOINT = BASE_URL + "/prediction"
SCORING_ENDPOINT = BASE_URL + "/scoring"
SUMMARY_STATS_ENDPOINT = BASE_URL + "/summarystats"
DIAGNOSTICS_ENDPOINT = BASE_URL + "/diagnostics"

# Define the dataset location
DATASET_LOCATION = "testdata/testdata.csv"

# Call each API endpoint and store the responses
response1 = requests.post(PREDICTION_ENDPOINT, params={"dataset_location": DATASET_LOCATION})
response2 = requests.get(SCORING_ENDPOINT, params={"dataset_location": DATASET_LOCATION})
response3 = requests.get(SUMMARY_STATS_ENDPOINT)
response4 = requests.get(DIAGNOSTICS_ENDPOINT)

# Combine all API responses
responses = {
    "prediction": response1.json(),
    "scoring": response2.json(),
    "summary_stats": response3.json(),
    "diagnostics": response4.text
}

# Write the combined responses to a file
with open("apireturns.txt", "w") as file:
    file.write(str(responses))

print("API responses have been written to apireturns.txt")
