import json
import pandas as pd

data_list = []
f_path = "IMC-Prosperity-2024/tutorial/logs/test-3-log.log"

# Function to try and parse a JSON string
def try_parse_json(json_string):
    try:
        return json.loads(json_string)  # Try to parse the JSON
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None

def parse_lambda_log(log_string):
    # Initialize an empty dictionary to hold the parsed data
    parsed_data = {}
    # Split the log string into parts based on newline
    parts = log_string.split('\n')
    for part in parts:
        if ': ' in part:
            key, value = part.split(': ', 1)
            # Directly assign parsed data into the dictionary
            parsed_data[key.strip()] = value.strip()
        # This approach assumes each line is a key-value pair split by ": "
    return parsed_data

# Open and read the log file
with open(f_path, "r") as file:
    json_str = ''  # Initialize an empty string to accumulate lines for a JSON object
    for line in file:
        # Check if the line is the start of a JSON object or part of an ongoing JSON object
        if line.startswith('{'):
            # If we're already accumulating a JSON object, we've reached a new one
            if json_str:
                # Try to parse the accumulated JSON string before resetting it
                data = try_parse_json(json_str)
                if data:  # If parsing was successful, add to our list
                    # Apply the lambdaLog parsing function here
                    parsed_lambda_log = parse_lambda_log(data['lambdaLog'])
                    data.update(parsed_lambda_log)  # Update the original data dict with parsed lambdaLog values
                    data_list.append(data)
                json_str = ''  # Reset the accumulator
            json_str += line  # Start accumulating lines for the new JSON object
        elif json_str:
            json_str += line  # Continue accumulating lines for the current JSON object

    # After the loop, try to parse any remaining JSON string
    if json_str:
        data = try_parse_json(json_str)
        if data:  # If parsing was successful, add to our list
            parsed_lambda_log = parse_lambda_log(data['lambdaLog'])
            data.update(parsed_lambda_log)  # Update with parsed lambdaLog values
            data_list.append(data)

# Convert the list of dictionaries into a DataFrame
df = pd.DataFrame(data_list)

# Convert numeric columns from strings to numeric types, handling any conversion errors
numeric_columns = ['Buy Order Depth', 'Sell Order Depth', 'Current Quantity', 'Acceptable price', 'Current Delta']
for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df.to_csv('current_log.csv')