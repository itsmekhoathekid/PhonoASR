
import json 

def load_json(file_path):
    """Load a JSON file and return its content as a dictionary."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def save_txt(file_path, content):
    """Save a string content to a text file."""
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

def process_json(json_paths):
    """Process the JSON data to extract and format text."""
    processed_text = []
    for path in json_paths:
        json_data = load_json(path)
        text_content = extract_text(json_data)
        processed_text.append(text_content)
    return '\n'.join(processed_text)

def extract_text(json_data):
    """Extract and format text from JSON data."""
    lines = []
    for item in json_data:
        lines.append(item["text"])
    return '\n'.join(lines)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python full_text_formation.py <output_txt_file> <input_json_file1> [<input_json_file2> ...]")
        sys.exit(1)

    output_txt_file = sys.argv[1]
    input_json_files = sys.argv[2:]

    full_text = process_json(input_json_files)
    save_txt(output_txt_file, full_text)