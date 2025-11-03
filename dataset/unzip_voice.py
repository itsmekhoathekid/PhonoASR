import sys
import zipfile
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to the compressed voice dataset (zip or rar)")
    parser.add_argument("--output", type=str, required=True, help="Directory to extract the dataset to")
    args = parser.parse_args()
    input_path = args.input
    output_path = args.output
    os.makedirs(output_path, exist_ok=True)
    if input_path.endswith('.zip'):
        print("Extracting zip file...")
        with zipfile.ZipFile(input_path, 'r') as zip_ref:
            zip_ref.extractall(output_path)
    else:
        print("Unsupported file format. Please provide a file.")
        sys.exit(1)
    print(f"Extracted {input_path} to {output_path}")