#!/bin/sh
#PBS -q A_002
#PBS -l select=1:ncpus=4:ngpus=1:mem=16gb
#PBS -M nghiangh@uit.edu.vn
#PBS -N conformer-training
#PBS -l walltime=24:00:00
#PBS -m be

source /home/inomar01/nlp@uit/PhonoASR/venv/bin/activate

python3 /home/inomar01/nlp@uit/PhonoASR/train.py --config /home/inomar01/nlp@uit/PhonoASR/configs/conformer-config.yaml
