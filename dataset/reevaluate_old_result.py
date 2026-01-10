import json
import re
import jiwer 

def load_json(json_path):
    """
    Load a json file and return the content as a dictionary.
    """
    with open(json_path, "r", encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_data(data, data_path):
    with open(data_path, "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def preprocess_skip_indexes(skip_indexes):
    """
    Preprocess the skip indexes to ensure they are integers.
    """
    processed_indexes = []
    for idx in skip_indexes:
        num = int(idx[2:])
        processed_indexes.append(num)
    return processed_indexes

def reevaluate_old_result(old_result_path, new_result_path, skip_indexs):
    # skip_indexs = preprocess_skip_indexes(skip_indexs)
    # print("Skip indexes:", skip_indexs[:10])
    old_res = load_json(old_result_path) 

    all_gold_text = []
    all_predicted_text = []
    print("Before processed length:", len(old_res))
    old_res = [item for idx, item in enumerate(old_res) if idx not in skip_indexs]
    print("After processed length:", len(old_res))
 
    for item in old_res[:-1]:
        pred = item['gold']
        target = item['predicted']

        all_gold_text.append(target)
        all_predicted_text.append(pred)
    
    wer = jiwer.wer(all_gold_text, all_predicted_text)
    cer = jiwer.cer(all_gold_text, all_predicted_text)
    print(f"Re-evaluated WER: {wer*100:.2f}%, CER: {cer*100:.2f}%")
    result = {
        "Re-evaluated WER": wer,
        "Re-evaluated CER": cer
    }
    # old_res.append({"Re-evaluated Result": result})
    # save_data(old_res, new_result_path)

    # rows = [x for x in old_res if isinstance(x, dict) and "WER" in x and "CER" in x]

    # wer_macro = sum(x["WER"] for x in rows) / len(rows)
    # cer_macro = sum(x["CER"] for x in rows) / len(rows)

    # print("macro WER%", wer_macro * 100)
    # print("macro CER%", cer_macro * 100)

old_result_path = './result/result-tasa-c2i-lsvsc.json'
new_result_path = './result/reevaluated-result-tasa-c2i-lsvsc.json'
skip_indexs = load_json("../dataset/LSVSC_test_unprocessed.json")['skip_indexes']

reevaluate_old_result(old_result_path, new_result_path, skip_indexs)
