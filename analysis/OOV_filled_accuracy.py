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

class oov_accuracy:
    def __init__(self, config, model_name): # config chua path cua result 
        self.config = config[model_name]
        

    def load_json(self, path):
        import json 
        with open(path, 'r') as f:
            data = json.load(f)
        return data


    def get_unique_tokens(self, data):
        """
        Get unique tokens from the dataset.
        """
        unique_tokens = set()
        for item in data:
            transcription = item["text"]
            for token in transcription.split():
                unique_tokens.add(token)
        return list(unique_tokens)
    

    def get_number_of_oov(self):
        
        vocab_train = self.load_json(self.config['vocab_word_path'])
        unique_token_test = self.get_unique_tokens(self.load_json(self.config['test_word_path']))
        oov_word = []
        for token in unique_token_test:
            if token not in vocab_train:
                oov_word.append(token)
        
        print(f"Number of OOV words: {len(oov_word)}, Percentage: {len(oov_word)/len(unique_token_test)*100:.2f}%")
        return oov_word, unique_token_test

    def save_json(self, data, path):
        import json 
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
    

    def calculate_oov_filled_accuracy(self, type_fill = "c2i"):

        test_result = self.load_json(self.config['result_test_word_path'])
        num_unk = 0
        idx_list_data = []

        for idx, item in enumerate(test_result):
            try:
                transcription = item["gold"]
                for idx_word, token in enumerate(transcription.split()):
                    if token == "<unk>":
                        num_unk += 1
                        idx_list_data.append((idx, idx_word))
            except: 
                continue
        
        
        test_result_type = self.load_json(self.config[f'result_{type_fill}_path'])
        
        test_data = self.load_json(self.config['result_test_word_path'])
        test_origin = self.load_json(self.config['test_word_path'])
        num_correct_fill = 0

        correct_filled = []
        cannot_be_filled = []
        tried_to_filled_but_wrong = []
        # test_data = self.load_json(self.config['test_word_path'])
        sample_correct = []
        total_num = 0
        for idx, idx_word in idx_list_data:
            if idx_word < len(test_result_type[idx]["predicted"].split(' ')) and idx_word < len(test_result_type[idx]["gold"].split(' ')):
                gold = test_result_type[idx]["gold"].split(' ')[idx_word]
                pred = test_result_type[idx]["predicted"].split(' ')[idx_word]
                # print(f"Gold: {gold}, Pred: {pred}")
                if gold == pred:
                    
                    correct_filled.append(gold)
                    num_correct_fill += 1
                    if test_result_type[idx]['gold'] == test_result_type[idx]["predicted"]:
                        print(f"Gold {type_fill}: {test_result_type[idx]['gold']} | pred {type_fill}: {test_result_type[idx]["predicted"]}")

                        sample_correct.append([
                            test_data[idx]['gold'],
                            test_origin[idx]['text']
                            
                        ])
                else:
                    cannot_be_filled.append(gold)
                    tried_to_filled_but_wrong.append(pred)
                total_num += 1
        oov_word, unique_token_test = self.get_number_of_oov()
        logging.info("OOV Filled Accuracy Results:")
        logging.info(f"Number of OOV words: {len(oov_word)}, Percentage: {len(oov_word)/len(unique_token_test)*100:.2f}%")
        logging.info(f"Number of <unk> in transcript of test dataset: {total_num}")
        logging.info(f"Number of correct fill by {type_fill} based model: {num_correct_fill}")
        logging.info(f"Percentage of correct fill by {type_fill} based model: {num_correct_fill/total_num*100:.2f}%")
        logging.info(f"Correctly filled words: {set(correct_filled)}")
        logging.info(f"Cannot be filled words: {set(cannot_be_filled)}")
        logging.info(f"Tried to fill but wrong predictions: {set(tried_to_filled_but_wrong)}")

        duplicate_check = set()
        for gold, text in sample_correct:
            if (gold, text) not in duplicate_check:
                logging.info(f"Gold unknown word: {gold} \nPredicted {type_fill} Converted: {text} \n{'-'*20}")
                duplicate_check.add((gold, text))
        
        if self.config['save']:
            self.save_json("./result/correctly_filled_words.json", list(set(correct_filled)))
            self.save_json("./result/cannot_be_filled_words.json", list(set(cannot_be_filled)))
            self.save_json("./result/tried_to_filled_but_wrong_predictions.json", list(set(tried_to_filled_but_wrong)))

        
import argparse
import yaml 

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description="Calculate OOV Filled Accuracy")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('--model_name', type=str, required=True, help='Model name in the config to use')
    parser.add_argument('--type_fill', type=str, default='oov_filled_accuracy.log', help='Path to the log file')
    args = parser.parse_args()

    config = load_config(args.config)

    log_file = os.path.join(config[args.model_name]['save_path'], args.type_fill)
    logg(log_file)

    oov_acc_calculator = oov_accuracy(config, args.model_name)
    oov_acc_calculator.calculate_oov_filled_accuracy(type_fill=args.type_fill)


if __name__ == "__main__":
    main()



