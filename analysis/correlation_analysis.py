from jiwer import process_words
import numpy as np
from scipy.stats import pearsonr, spearmanr

import logging
import os 

def logg(log_file):
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # vẫn in ra màn hình
        ]
    )

class correlation_analysis:
    def __init__(self, config):
        self.config = config 
    

    def load_json(self, path):
        import json 
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    
    def save_json(self, data, path):
        import json 
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
    
    def compute_token_stats_with_jiwer(self, path, correct_at_least=1, save_path=None):
        data = self.load_json(path)
        data = [x for x in data if isinstance(x, dict) and "gold" in x and "predicted" in x]

        stats = {}

        for item in data:
            gold = item["gold"].strip()
            pred = item["predicted"].strip()

            out = process_words(gold, pred)

            aligns = out.alignments[0]

            # unwrap references/hypotheses if nested
            ref = out.references[0]     # list[str]

            for ch in aligns:
                op = ch.type
                rs, re = ch.ref_start_idx, ch.ref_end_idx

                if op == "insert":
                    continue

                # for equal/delete/substitute: total increases for ref tokens
                for tok in ref[rs:re]:
                    # tok must be a string token
                    if isinstance(tok, list):
                        # unexpected nesting; flatten
                        for t in tok:
                            stats.setdefault(t, {"total": 0, "correct": 0})
                            stats[t]["total"] += 1
                            if op == "equal":
                                stats[t]["correct"] += 1
                    else:
                        stats.setdefault(tok, {"total": 0, "correct": 0})
                        stats[tok]["total"] += 1
                        if op == "equal":
                            stats[tok]["correct"] += 1

        unique_correct = sum(1 for tok, s in stats.items() if s["correct"] >= correct_at_least)

        logging.info(f"Number of unique tokens correctly predicted at least {correct_at_least} times: {unique_correct}")
        logging.info(f"Total unique tokens in GT: {len(stats)}")


        # stats = {tok: s for tok, s in stats.items() if s["correct"] >= correct_at_least}
        if save_path:
            self.save_json(stats, save_path)
        
        return stats

    def word_frequency_training_set(self, path, save_path = None, type = 'phoneme'):
        data = self.load_json(path)
        word_count = {}
        for item in data:
            
            transcription = item["phoneme_presentation"] if type == 'phoneme' else item["text"]
            for token in transcription.split(' '):
                if token not in word_count:
                    word_count[token] = 0
                word_count[token] += 1
        if save_path:
            self.save_json(word_count, save_path)
        return word_count
    

    

    def build_vectors(self, test_stats, train_freq, include_zero=False):
        freqs, recalls = [], []

        for w, s in test_stats.items():
            if s["total"] == 0:
                continue

            recall = s["correct"] / s['total']
            freq = train_freq.get(w, 0)
            

            if not include_zero and freq == 0:
                continue

            freqs.append(freq)
            recalls.append(recall)

        return np.array(freqs), np.array(recalls)
    
    def compute_correlation(self, type_fill = 'word', correct_at_least=1):
        assert type_fill in ['word', 'phoneme', 'char'], "type_fill must be one of 'word', 'phoneme', or 'char'"
        unique_word_count_test_set = self.compute_token_stats_with_jiwer(self.config[f'result_{type_fill}'], correct_at_least=correct_at_least)
        unique_word_count_training_set = self.word_frequency_training_set(self.config[f'train_{type_fill}'], type = type_fill)

        freqs, recalls = self.build_vectors(
            unique_word_count_test_set,
            unique_word_count_training_set,
            include_zero=True   # giống paper
        )

        pearson_r, _ = pearsonr(freqs, recalls)
        spearman_rho, _ = spearmanr(freqs, recalls)

        logging.info(f"Correlation analysis for {type_fill} level:")
        logging.info(f"Words used: {len(freqs)}")
        logging.info(f"Pearson r: {pearson_r}")
        logging.info(f"Spearman rho: {spearman_rho}")

def load_config(path):
    import yaml
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the config file."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="tasa-w2i-lsvsc",
        help="Model name in the config to use."
    )
    parser.add_argument(
        "--type_fill",
        type=str,
        default="word",
        help="Type of fill: 'word' or 'phoneme'"
    )
    
    args = parser.parse_args()
    config = load_config(args.config)

    logg(config[args.model_name]['log_file'])

    analysis = correlation_analysis(config=config[args.model_name])
    analysis.compute_correlation(type_fill=args.type_fill)
