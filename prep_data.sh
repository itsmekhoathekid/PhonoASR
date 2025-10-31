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

set -e

mkdir -p dataset
cd dataset

pip install gdown librosa speechbrain jiwer

if [[ "$2" == "vivos" ]]; then
    echo "Downloading VIVOS dataset..."
    gdown 19CV4WZgYez-i2oHV2r9maJofjNqcTX4o
    gdown 1v75mLO-TVfPXe27o54JMlXD5cQ81eaVG
    gdown 1YgTF-NbHuweHWr2LahS_X9j--laGDnIK
    base_wav_path=$(pwd)/voices
    python /datastore/npl/Speech2Text/PhonoASR/unzip_voice.py --input "./voices.zip" --output "./voices"

elif [[ "$2" == "commonvoice" ]]; then
    echo "Downloading Common Voice dataset..."
    gdown 189KSe2sSBD8Y3vaVZKZuU1Ca6OHvJ3DD
    gdown 1VtdWkozPypFaZP5pty_ID3hSQyr75Mt0
    gdown 1fM54Z9VCTVzTmib_KvGqM5GpCS8deW0V
    gdown 1vVjQCCMvVZvButmMquAKsTMx_FVHshh-
    base_wav_path=$(pwd)/clips
    python /datastore/npl/Speech2Text/PhonoASR/unzip_voice.py --input "./voices.zip" --output "./voices"
    

elif [[ "$2" == "vietmed" ]]; then
    echo "Downloading VietMed dataset..."
    gdown 1hTVAZXY3kdfCJVSzUZuhzS3U7SFB-xHp
    gdown 1IPbjiHUCBvUgQ_k2PRz9vYJdJLiZM2aC
    gdown 1WVe0yHlCuMyEuvR9huJdatOwrpA5-njr
    gdown 1vo7jF2JKpiJ3w5OfKW5f4jypO8e9q5hk
    base_wav_path=$(pwd)/wav
    python /datastore/npl/Speech2Text/PhonoASR/unzip_voice.py --input "./wav.zip" --output "./wav"

elif [[ "$2" == "lsvsc" ]]; then
    echo "Downloading LSVSC dataset..."
    gdown 1ADHw6xJOOWi3HDTtaqr5HLlr6hAxdouQ
    gdown 1wOvsfHnX1TDJgLatba5WoqIQ99kd0r_g
    gdown 1XKJRIf32tD0f8hUk4sdSl_E1eYkALG8d
    gdown 1kDBJv-aym-dnK6ztMikBXnZqyJPMsxS4
    base_wav_path=$(pwd)/LSVSC_100/data
    python /datastore/npl/Speech2Text/PhonoASR/unzip_voice.py --input "./LSVSC_100.zip" --output "./LSVSC_100"
    
fi

base_path=$(pwd)

cd /

if [[ "$1" == "phoneme" ]]; then
    echo "Preprocessing for phoneme-based model"

    if [[ "$2" == "vietmed" ]]; then
        train_path="/datastore/npl/Speech2Text/dataset/labeled_medical_data_train_transcript.json"
        test_path="/datastore/npl/Speech2Text/dataset/labeled_medical_data_test_transcript.json"
    elif [[ "$2" == "lsvsc" ]]; then
        train_path="/datastore/npl/Speech2Text/dataset/LSVSC_train.json"
        test_path="/datastore/npl/Speech2Text/dataset/LSVSC_test.json"
        valid_path="/datastore/npl/Speech2Text/dataset/LSVSC_valid.json"
    else
        train_path="/datastore/npl/Speech2Text/dataset/train.json"
        test_path="/datastore/npl/Speech2Text/dataset/test.json"
    fi

    python /datastore/npl/Speech2Text/PhonoASR/dataset/mutiple_construct.py \
        --dataset "$2" \
        --type_tokenizer "phoneme" \
        --train_path "$train_path" \
        --test_path "$test_path" \
        --base_wav_path "$base_wav_path" \
        --base_path "$base_path" \
        --valid_path "$valid_path"
        
    
elif [[ "$1" == "char" ]]; then
    echo "Preprocessing for normal model"
    python /datastore/npl/Speech2Text/PhonoASR/dataset/mutiple_construct.py \
        --dataset "$2" \
        --type_tokenizer "char" \
        --train_path "/datastore/npl/Speech2Text/dataset/train.json" \
        --test_path "/datastore/npl/Speech2Text/dataset/test.json" \
        --base_wav_path "$base_wav_path" \
        --base_path "$base_path"

else
    echo "Preprocessing for normal model"
    python /datastore/npl/Speech2Text/PhonoASR/dataset/mutiple_construct.py \
        --dataset "$2" \
        --type_tokenizer "word" \
        --train_path "/datastore/npl/Speech2Text/dataset/train.json" \
        --test_path "/datastore/npl/Speech2Text/dataset/test.json" \
        --base_wav_path "$base_wav_path" \
        --base_path "$base_path"
fi
mkdir -p /datastore/npl/Speech2Text/PhonoASR/saves