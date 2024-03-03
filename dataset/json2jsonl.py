import sys
import json

def json_to_jsonl(input_json_file, output_jsonl_file):
    with open(input_json_file, 'r') as json_file, open(output_jsonl_file, 'w') as jsonl_file:
        # Load the entire JSON file
        data = json.load(json_file)

        # Write each JSON object as a separate line in the JSON Lines file
        for item in data:
            jsonl_file.write(json.dumps(item) + '\n')

# Example usage
input_json_file = sys.argv[1]  # Replace with your input JSON file
output_jsonl_file = input_json_file + "l"

json_to_jsonl(input_json_file, output_jsonl_file)
