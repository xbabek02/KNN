#!/bin/bash
#PBS -N batch_job_example
#PBS -l select=1:ncpus=2:ngpus=1:mem=32gb:scratch_local=80gb: 
#PBS -l walltime=20:00:00

module add python/3.11.11-gcc-10.2.1-555dlyc

python -m ensurepip --upgrade

# Set the TMPDIR environment variable globally
export TMPDIR=$SCRATCHDIR
export HOME=/storage/brno2/home/mutanin

python -m pip install torch transformers pandas numpy matplotlib seaborn scikit-learn Pillow tqdm python-Levenshtein editdistance torchvision torchaudio kagglehub tensorboard wordcloud

cd /storage/brno2/home/mutanin/KNN

# python script.py

clean_scratch
