import os
from pydub import AudioSegment
from concurrent.futures import ProcessPoolExecutor, as_completed
import ast
import json

def convert_single_ogg_to_wav(ogg_path, wav_path):
    """Hàm convert 1 file"""
    audio = AudioSegment.from_ogg(ogg_path)
    audio.export(wav_path, format="wav")
    print(f"✅ Converted: {ogg_path} -> {wav_path}")

def convert_folder_ogg_to_wav(input_folder_path, output_folder, max_workers=8):
    """Convert toàn bộ file .ogg sang .wav bằng multiprocessing"""
    os.makedirs(output_folder, exist_ok=True)
    tasks = []

    # gom tất cả các file .ogg cần convert
    for folder in os.listdir(input_folder_path):
        folder_path = os.path.join(input_folder_path, folder)
        if not os.path.isdir(folder_path):
            continue
        for ogg_file in os.listdir(folder_path):
            if ogg_file.endswith(".ogg"):
                ogg_path = os.path.join(folder_path, ogg_file)
                wav_name = ogg_file.replace(".ogg", ".wav")
                wav_path = os.path.join(output_folder, wav_name)
                tasks.append((ogg_path, wav_path))

    # chạy song song
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(convert_single_ogg_to_wav, ogg, wav) for ogg, wav in tasks]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print(f"❌ Error: {e}")

# if __name__ == "__main__":
#     convert_folder_ogg_to_wav(
#         input_folder_path="/mnt/d/labeled_medical_data/test_audio/test_audio",
#         output_folder="/mnt/d/labeled_medical_data/dev_audio/wav",
#         max_workers=8  # tùy CPU
#     )

def save_json(data, json_path):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def construct_json(transcript_path):
    with open(transcript_path, "r", encoding="utf-8") as f:
        data_str = f.read().strip()

    data_list = ast.literal_eval(data_str)
    count = 0
    res = {}
    for item in data_list:
        data_res = {}
        name = item['file'].split("/")[-1].replace(".ogg", ".wav")
        data_res['voice'] = name
        data_res['script'] = item['text']
        res[name.replace(".wav","")] = data_res
        count += 1
    print(f"Total records processed: {count}")
    print("Saveing to JSON...")
    save_json(res, transcript_path.replace(".txt", ".json"))

paths = ['/mnt/d/labeled_medical_data/labeled_medical_data_test_transcript.txt',
         '/mnt/d/labeled_medical_data/labeled_medical_data_dev_transcript.txt',
         '/mnt/d/labeled_medical_data/labeled_medical_data_train_transcript.txt']

for path in paths:
    construct_json(path)