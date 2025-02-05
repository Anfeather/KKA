import json

# Load data from flower.txt
input_file_path = 'flower.txt'
with open(input_file_path, 'r', encoding='utf-8') as file:
    captions = file.readlines()

# Clean and prepare the data
captions = [caption.strip() for caption in captions]

# Convert each caption into a nested list format [["caption"], ["caption"], ["caption"], ["caption"], ["caption"]]
output_data = [caption for caption in captions for _ in range(10)]

# Define the output path for the new JSON file
output_file_path = 'text_caption.json'

# Write the JSON data to file
with open(output_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(output_data, json_file, ensure_ascii=False, indent=4)

print(f"JSON data successfully written to {output_file_path}")

