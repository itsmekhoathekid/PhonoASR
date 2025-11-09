import json
import re
from Vietnamese_utils import analyse_Vietnamese
import os

class DatasetPreparing:
    def __init__(self, dataset_name, base_wav_path, type_tokenizer="word"):
        self.dataset = dataset_name
        self.base_wav_path = base_wav_path
        self.type_tokenizer = type_tokenizer

    def normalize_transcript(self, text):
        text = text.lower()
        text = re.sub(r"[\'\"(),.!?]:", " ", text)
        text = re.sub(r"\s+", " ", text)  # loại bỏ khoảng trắng dư
        return text.strip()
    
    def load_json(self, json_path):
        """
        Load a json file and return the content as a dictionary.
        """
        with open(json_path, "r", encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def save_data(self, data, data_path):
        with open(data_path, "w", encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def create_vocab(self, json_path, wrong2correct, dataset, vocab_path):
        unprocsssed = []
        data = self.load_json(json_path)

        vocab = {
            "<pad>": 0,
            "<s>": 1,
            "</s>": 2,
            "<unk>": 3,
            "<space>": 4,
            "<blank>" : 5
        }

        for idx, item in data.items():
            if dataset == "vivos" or dataset == "vietmed":
                text = self.normalize_transcript(item['script'])
            elif dataset == "commonvoice":
                text = self.normalize_transcript(item['transcript'])
            elif dataset == "lsvsc":
                text = self.normalize_transcript(item['text'])
            

            if self.type_tokenizer == "word":
                for word in text.split():
                    if word not in vocab:
                        vocab[word] = len(vocab)
            elif self.type_tokenizer == "char":
                for word in text.strip():
                    if word not in vocab:
                        vocab[word] = len(vocab)
            elif self.type_tokenizer == "phoneme":
                for word in text.split():
                    try:
                        initial, rhyme, tone = analyse_Vietnamese(word)
                        if initial not in vocab:
                            vocab[initial] = len(vocab)
                        if rhyme not in vocab:
                            vocab[rhyme] = len(vocab)
                        if tone not in vocab:
                            vocab[tone] = len(vocab)
                    except:
                        if word in wrong2correct.keys():
                            correct_word = wrong2correct[word]
                            try:
                                initial, rhyme, tone = analyse_Vietnamese(correct_word)
                                if initial not in vocab:
                                    vocab[initial] = len(vocab)
                                if rhyme not in vocab:
                                    vocab[rhyme] = len(vocab)
                                if tone not in vocab:
                                    vocab[tone] = len(vocab)
                            except:
                                unprocsssed.append(word)
        self.save_data(vocab, vocab_path)    
        print(f"Vocabulary saved to {vocab_path}")
        return vocab, list(set(unprocsssed))
    
    def process_data(self, data_path, vocab, default_data_path, save_path, dataset = "vivos", type = "stack"):
        data = self.load_json(data_path)


        res = []
        for idx, item in data.items():
            
            data_res = {}
            if dataset == "vivos" or dataset == "vietmed":
                text = self.normalize_transcript(item['script'])
            elif dataset == "commonvoice":
                text = self.normalize_transcript(item['transcript'])
            elif dataset == "lsvsc":
                text = self.normalize_transcript(item['text'])
            

            if self.type_tokenizer == "word":
                unk_id = vocab["<unk>"]
                tokens = [vocab.get(word, unk_id) for word in text.split()]
                data_res['encoded_text'] = tokens
                data_res['text'] = text
                # data_res['wav_path'] = os.path.join(default_data_path, item['voice'])
            
            elif self.type_tokenizer == "char":
                unk_id = vocab["<unk>"]
                tokens = [vocab.get(word, unk_id) for word in text.strip()]
                data_res['encoded_text'] = tokens
                data_res['text'] = text
                # data_res['wav_path'] = os.path.join(default_data_path, item['voice'])
            
            elif self.type_tokenizer == "phoneme":
                unk_id = vocab["<unk>"]
                # tokens = [vocab.get(word, unk_id) for word in text.split()]

                tokens = []
                for word in text.split():
                    try:
                        initial, rhyme, tone = analyse_Vietnamese(word)
                        word_list = [vocab.get(initial, unk_id), vocab.get(rhyme, unk_id), vocab.get(tone, unk_id)]
                        if type == "stack":
                            # print("hi")
                            tokens.append(word_list)
                        else:
                            # print("hi")
                            tokens += word_list
                            tokens += [vocab["<space>"]]
                    except:
                        continue


                data_res['encoded_text'] = tokens[:-1] if type != "stack" else tokens
                data_res['text'] = text
            data_res['wav_path'] = os.path.join(default_data_path, item['voice']) if dataset != "lsvsc" else os.path.join(default_data_path, item['wav'])
            res.append(data_res)
        self.save_data(res, save_path)
        print(f"Data saved to {save_path}")


wrong2correct = {
    "piêu": "phiêu",
    "quỉ": "quỷ",
    "téc": "tét",
    "quoạng": "quạng",
    "đéc": "đét",
    "quĩ": "quỹ",
    "ka": "ca",
    "gen": "ghen",
    "qui": "quy",
    "ngía": "nghía",
    "quít": "quýt",
    "yêng": "yên",
    "séc": "sét",
    "quí": "quý",
    "quị": "quỵ",
    "pa": "ba",
    "ko": "không",
    "léc": "lét",
    "pí": "bí",
    "quì": "quỳ",
    "pin": "bin"
}

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="default", help="Dataset type: default, commonvoice, vivos")
parser.add_argument("--type_tokenizer", type=str, default="word", help="Type of tokenizer: word, char, phoneme")
parser.add_argument("--train_path", type=str, required=True, help="Path to the input data json file")
parser.add_argument("--test_path", type=str, required=True, help="Path to save the processed data json file")
parser.add_argument("--valid_path", type=str, required=False, help="Path to save the vocabulary json file")
parser.add_argument("--base_wav_path", type=str, required=True, help="Base path to the wav files")
parser.add_argument("--base_path", type=str, required=True, help="Base path to the wav files")
args = parser.parse_args()

dataset = args.dataset
data_preparer = DatasetPreparing(dataset_name=dataset, base_wav_path=args.base_wav_path, type_tokenizer=args.type_tokenizer)
vocab, unprocossed = data_preparer.create_vocab(args.train_path, wrong2correct, dataset, os.path.join(args.base_path, f"{args.type_tokenizer}_vocab_{args.dataset}.json"))

data_preparer.process_data(args.train_path,
             vocab,
             args.base_wav_path,
             args.train_path.replace(".json", f"_{args.type_tokenizer}.json"), dataset)

data_preparer.process_data(args.test_path,
             vocab,
             args.base_wav_path,
             args.test_path.replace(".json", f"_{args.type_tokenizer}.json"), dataset)

if args.valid_path:
    data_preparer.process_data(args.valid_path,
             vocab,
             args.base_wav_path,
             args.valid_path.replace(".json", f"_{args.type_tokenizer}.json"), dataset)