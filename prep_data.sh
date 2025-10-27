#!/bin/bash

if [[ "$1" == "--help" ]]; then
    echo "How to: $0 [option]"
    echo "  --help     Instructions"
    echo "  phoneme    Preprocess for phoneme-based model"
    echo "  normal     Preprocess for normal model (default)"
    exit 0
elif [[ -z "$1" ]]; then
    echo "No option specified. Defaulting to normal model preprocessing."
    set -- "normal"
fi

if [[ "&2" == "--help" ]]; then
    echo "How to: $0 [option]"
    echo "  --help     Instructions"
    echo "  vivos    Processing vivos dataset"
    echo "  commonvoice     Processing commonvoice dataset"
    exit 0
elif [[ -z "$2" ]]; then
    echo "No option specified. IDK what to do man."
    exit 1

elif [[ "$2" != "vivos" && "$2" != "commonvoice" ]]; then
    echo "Invalid second argument. Use --help for instructions."
    exit 1
fi

set -e

mkdir -p dataset
cd dataset

pip install gdown librosa speechbrain jiwer
if [[ "$2" == "vivos" ]]; then
    echo "Downloading VIVOS dataset..."
    gdown 19CV4WZgYez-i2oHV2r9maJofjNqcTX4o 
    gdown 1v75mLO-TVfPXe27o54JMlXD5cQ81eaVG 
    gdown 1YgTF-NbHuweHWr2LahS_X9j--laGDnIK
    unzip -o voices.zip
    base_wav_path=$(pwd)/voices

elif [[ "$2" == "commonvoice" ]]; then
    echo "Downloading Common Voice dataset..."
    gdown 189KSe2sSBD8Y3vaVZKZuU1Ca6OHvJ3DD
    gdown 1VtdWkozPypFaZP5pty_ID3hSQyr75Mt0
    gdown 1fM54Z9VCTVzTmib_KvGqM5GpCS8deW0V
    gdown 1vVjQCCMvVZvButmMquAKsTMx_FVHshh-
    unzip -o clips.zip
    base_wav_path=$(pwd)/clips
fi


cd /
if [[ "$1" == "phoneme" ]]; then
    echo "Preprocessing for phoneme-based model"
    python workspace/PhonoASR/dataset/mutiple_construct.py --dataset "$2" --type_tokenizer "phoneme" --train_path "workspace/dataset/train.json" --test_path "workspace/dataset/test.json" --base_wav_path "$base_wav_path"
elif [[ "$1" == "char" ]]; then
    echo "Preprocessing for normal model"
    python workspace/PhonoASR/dataset/mutiple_construct.py --dataset "$2" --type_tokenizer "char" --train_path "workspace/dataset/train.json" --test_path "workspace/dataset/test.json" --base_wav_path "$base_wav_path"
else
    echo "Preprocessing for normal model"
    python workspace/PhonoASR/dataset/mutiple_construct.py --dataset "$2" --type_tokenizer "word" --train_path "workspace/dataset/train.json" --test_path "workspace/dataset/test.json" --base_wav_path "$base_wav_path"
fi
mkdir workspace/PhonoASR/saves