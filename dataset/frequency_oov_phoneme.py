import json
from Vietnamese_utils import *

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

from tqdm import tqdm
def phoneme_frequency_in_path(train_path):
    res = {}
    data = load_json(train_path)
    for idx, item in tqdm(data.items(), desc = "processing"):
        text = item['text']
        for word in text.split(' '):
            # print(word)
            initial, rhyme, tone = analyse_Vietnamese(word)
            if initial not in res:
                res[initial] = 1
            else:
                res[initial] += 1
            if rhyme not in res:
                res[rhyme] = 1
            else:
                res[rhyme] += 1
            if tone not in res:
                res[tone] = 1
            else:
                res[tone] += 1  
    
    return res


def get_component_frequency(oov_path, train_path, save_path=None):
    oov_data = load_json(oov_path)
    phoneme_freq = phoneme_frequency_in_path(train_path)

    component_freq = {}
    for item in oov_data['oov_words']:
        initial, rhyme, tone = analyse_Vietnamese(item)
        component_freq[initial] = phoneme_freq.get(initial, 0)
        component_freq[rhyme] = phoneme_freq.get(rhyme, 0)
        component_freq[tone] = phoneme_freq.get(tone, 0)
    
    if save_path:
        save_data(component_freq, save_path)
    
    return component_freq


oov_path = '/home/anhkhoa/PhonoASR/saves/oov_words.json'
train_path = '/home/anhkhoa/dataset/LSVSC_train.json'
save_path = '/home/anhkhoa/PhonoASR/saves/lsvsc_phoneme_frequency.json'

get_component_frequency(oov_path, train_path, save_path)

