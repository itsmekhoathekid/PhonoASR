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

elif [[ "$2" != "vivos" && "$2" != "commonvoice"  && "$2" != "vietmed" ]]; then
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
elif [[ "$2" == "vietmed" ]]; then
    echo "Downloading VietMed dataset..."
    gdown 1hTVAZXY3kdfCJVSzUZuhzS3U7SFB-xHp
    gdown 1IPbjiHUCBvUgQ_k2PRz9vYJdJLiZM2aC
    gdown 1WVe0yHlCuMyEuvR9huJdatOwrpA5-njr
    gdown 1vo7jF2JKpiJ3w5OfKW5f4jypO8e9q5hk
    base_wav_path=$(pwd)/wav
    unzip -o wav.zip
elif [[ "$2" == "lsvsc" ]]; then
    echo "Downloading LSVSC dataset..."
    gdown 1EcyKmU_-TnoEqyV9FFSLMyyDqIn2pa2N
    gdown 1bX646Tu_geOhoCZjn6l_4ohUgga18Qw_
    gdown 1IChYihRLp7O2cRFIpOi8jEvLnrySm1Bf
    gdown 1bTLibQ8rmXo82YXViUr7wc2JZ5oIg9xQ
    base_wav_path=$(pwd)/LSVSC_100
    unzip -o LSVSC_100.rar
fi
base_path=$(pwd)

cd /
if [[ "$1" == "phoneme" ]]; then
    echo "Preprocessing for phoneme-based model"

    if [[ "$2" == "vietmed" ]]; then
        train_path="workspace/dataset/labeled_medical_data_train_transcript.json"
        test_path="workspace/dataset/labeled_medical_data_test_transcript.json"
    elif [[ "$2" == "lsvsc" ]]; then
        train_path="workspace/dataset/LSVSC_train.json"
        test_path="workspace/dataset/LSVSC_test.json"
        valid_path="workspace/dataset/LSVSC_valid.json"
    else
        train_path="workspace/dataset/train.json"
        test_path="workspace/dataset/test.json"
    fi

    python workspace/PhonoASR/dataset/mutiple_construct.py \
        --dataset "$2" \
        --type_tokenizer "phoneme" \
        --train_path "$train_path" \
        --test_path "$test_path" \
        --base_wav_path "$base_wav_path" \
        --base_path "$base_path" \
        --valid_path "$valid_path"
        
    
elif [[ "$1" == "char" ]]; then
    echo "Preprocessing for normal model"
    python workspace/PhonoASR/dataset/mutiple_construct.py --dataset "$2" --type_tokenizer "char" --train_path "workspace/dataset/train.json" --test_path "workspace/dataset/test.json" --base_wav_path "$base_wav_path" --base_path "$base_path"
else
    echo "Preprocessing for normal model"
    python workspace/PhonoASR/dataset/mutiple_construct.py --dataset "$2" --type_tokenizer "word" --train_path "workspace/dataset/train.json" --test_path "workspace/dataset/test.json" --base_wav_path "$base_wav_path" --base_path "$base_path"
fi
mkdir workspace/PhonoASR/saves