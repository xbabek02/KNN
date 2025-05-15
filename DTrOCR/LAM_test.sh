#!/bin/bash
#PBS -N batch_job_example
#PBS -l select=1:ncpus=2:ngpus=1:mem=32gb:scratch_local=80gb:gpu_mem=40gb 
#PBS -l walltime=20:00:00

# Load Python module
module add python/3.11.11-gcc-10.2.1-555dlyc

python -m ensurepip --upgrade

# Set the TMPDIR environment variable globally
export TMPDIR=$SCRATCHDIR
export HOME=/storage/brno2/home/xbabek02

python -m pip install torch torchvision torchaudio pandas numpy matplotlib seaborn scikit-learn Pillow tensorboard tqdm kagglehub python-Levenshtein editdistance

cd
cd KNN/DTrOCR

python LAM_test.py --exp-name TEST --eval-iter 10 --load-model output/LAM/LAM_finetuned.pth

clean_scratch
