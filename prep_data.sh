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

if [[ "$2" == "--help" ]]; then
    echo "How to: $0 [option]"
    echo "  --help          Instructions"
    echo "  vivos           Processing vivos dataset"
    echo "  commonvoice     Processing commonvoice dataset"
    echo "  vietmed         Processing vietmed dataset"
    echo "  lsvsc           Processing lsvsc dataset"
    exit 0
elif [[ -z "$2" ]]; then
    echo "No dataset specified."
    exit 1
elif [[ "$2" != "vivos" && "$2" != "commonvoice" && "$2" != "vietmed" && "$2" != "lsvsc" ]]; then
    echo "Invalid second argument. Use --help for instructions."
    exit 1
fi

DATA_DIR="$(pwd)"

set -e

echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install gdown librosa speechbrain jiwer
pip install git+https://github.com/lhotse-speech/lhotse
pip install https://huggingface.co/csukuangfj/k2/resolve/main/ubuntu-cuda/k2-1.24.4.dev20250807+cuda12.8.torch2.8.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl

if [[ -d "dataset" ]]; then
    echo "Folder 'dataset' already exists. Moving into it..."
    cd dataset
else
    echo "Folder 'dataset' not found. Creating new one..."
    mkdir dataset
    cd dataset
fi


if [[ "$2" == "vivos" ]]; then
    echo "Downloading VIVOS dataset..."
    gdown 19CV4WZgYez-i2oHV2r9maJofjNqcTX4o
    gdown 1v75mLO-TVfPXe27o54JMlXD5cQ81eaVG
    gdown 1YgTF-NbHuweHWr2LahS_X9j--laGDnIK
    base_wav_path=$(pwd)/voices
    base_path=$(pwd)
    cd ..
    python ./PhonoASR/dataset/unzip_voice.py --input "./voices.zip" --output "./voices"

elif [[ "$2" == "commonvoice" ]]; then
    echo "Downloading Common Voice dataset..."
    gdown 189KSe2sSBD8Y3vaVZKZuU1Ca6OHvJ3DD
    gdown 1VtdWkozPypFaZP5pty_ID3hSQyr75Mt0
    gdown 1fM54Z9VCTVzTmib_KvGqM5GpCS8deW0V
    gdown 1vVjQCCMvVZvButmMquAKsTMx_FVHshh-
    base_wav_path=$(pwd)/clips
    base_path=$(pwd)
    cd ..
    python ./PhonoASR/dataset/unzip_voice.py --input "./dataset/clips.zip" --output "./dataset/clips"
    

elif [[ "$2" == "vietmed" ]]; then
    echo "Downloading VietMed dataset..."
    gdown 1hTVAZXY3kdfCJVSzUZuhzS3U7SFB-xHp
    gdown 1IPbjiHUCBvUgQ_k2PRz9vYJdJLiZM2aC
    gdown 1WVe0yHlCuMyEuvR9huJdatOwrpA5-njr
    gdown 1vo7jF2JKpiJ3w5OfKW5f4jypO8e9q5hk
    base_wav_path=$(pwd)/wav
    base_path=$(pwd)
    cd ..
    python ./PhonoASR/dataset/unzip_voice.py --input "./dataset/wav.zip" --output "./dataset/wav"

elif [[ "$2" == "lsvsc" ]]; then
    echo "Downloading LSVSC dataset..."
    gdown 1ADHw6xJOOWi3HDTtaqr5HLlr6hAxdouQ
    gdown 1wOvsfHnX1TDJgLatba5WoqIQ99kd0r_g
    gdown 1XKJRIf32tD0f8hUk4sdSl_E1eYkALG8d
    gdown 1kDBJv-aym-dnK6ztMikBXnZqyJPMsxS4
    base_wav_path=$(pwd)/LSVSC_100/data
    base_path=$(pwd)
    cd ..
    python ./PhonoASR/dataset/unzip_voice.py --input "./dataset/LSVSC_100.zip" --output "./dataset/LSVSC_100"
    
fi

if [[ "$2" == "vietmed" ]]; then
    train_path="$DATA_DIR/dataset/labeled_medical_data_train_transcript.json"
    test_path="$DATA_DIR/dataset/labeled_medical_data_test_transcript.json"
    valid_path="$DATA_DIR/dataset/labeled_medical_data_dev_transcript.json"

elif [[ "$2" == "lsvsc" ]]; then
    train_path="$DATA_DIR/dataset/LSVSC_train.json"
    test_path="$DATA_DIR/dataset/LSVSC_test.json"
    valid_path="$DATA_DIR/dataset/LSVSC_valid.json"
else
    train_path="$DATA_DIR/dataset/train.json"
    test_path="$DATA_DIR/dataset/test.json"
fi

if [[ "$1" == "phoneme" ]]; then
    echo "Preprocessing for phoneme-based model"

    python ./PhonoASR/dataset/mutiple_construct.py \
        --dataset "$2" \
        --type_tokenizer "phoneme" \
        --train_path "$train_path" \
        --test_path "$test_path" \
        --base_wav_path "$base_wav_path" \
        --base_path "$base_path" \
        --valid_path "$valid_path"

    python ./PhonoASR/dataset/check_empty.py \
        --input "${train_path%.json}_phoneme.json" \

    python ./PhonoASR/dataset/check_empty.py \
        --input "${test_path%.json}_phoneme.json" \

    if [[ -n "$valid_path" ]]; then
        python ./PhonoASR/dataset/check_empty.py \
            --input "${valid_path%.json}_phoneme.json"
    fi

elif [[ "$1" == "char" ]]; then
    echo "Preprocessing for normal model"
    python ./PhonoASR/dataset/mutiple_construct.py \
        --dataset "$2" \
        --type_tokenizer "char" \
        --train_path "$train_path" \
        --test_path "$test_path" \
        --base_wav_path "$base_wav_path" \
        --base_path "$base_path"

    python ./PhonoASR/dataset/check_empty.py \
        --input "${train_path%.json}_char.json" \

    python ./PhonoASR/dataset/check_empty.py \
        --input "${test_path%.json}_char.json" \

    if [[ -n "$valid_path" ]]; then
        python ./PhonoASR/dataset/check_empty.py \
            --input "${valid_path%.json}_char.json"
    fi

else
    echo "Preprocessing for normal model"
    python ./PhonoASR/dataset/mutiple_construct.py \
        --dataset "$2" \
        --type_tokenizer "word" \
        --train_path "$train_path" \
        --test_path "$test_path" \
        --base_wav_path "$base_wav_path" \
        --base_path "$base_path"

    python ./PhonoASR/dataset/check_empty.py \
        --input "${train_path%.json}_word.json" \

    python ./PhonoASR/dataset/check_empty.py \
        --input "${test_path%.json}_word.json" \

    if [[ -n "$valid_path" ]]; then
        python ./PhonoASR/dataset/check_empty.py \
            --input "${valid_path%.json}_word.json"
    fi

fi