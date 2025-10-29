import json 
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to the input JSON file")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered_data = [item for item in data if item.get("encoded_text")]

    with open(args.input, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=4)

    print(f"Done! {len(data) - len(filtered_data)} items removed.")