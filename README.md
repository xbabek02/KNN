# KNN - Handwritten text recognition (OCR)



**Členové týmu:**
- Přemek Janda (xjanda28)
- Petr Volf (xvolfp00)
- Radomír Bábek (xbabek02)


## Spuštění 
```sh
cd DTrOCR
```

### trénování, testování, finetuning
testování:
```sh 
python test.py --exp-name TEST/LAM_finetuned_IAM_dataset --eval-iter 1000000 --print-iter 0 --load-model /storage/brno2/home/xbabek02/LAM_finetuned.pth --dataset-name iam --seed 42 --plot-eval
python test.py --exp-name TEST/LAM_finetuned_LAM_dataset --eval-iter 1000000 --print-iter 0 --load-model /storage/brno2/home/xbabek02/LAM_finetuned.pth --dataset-name lam --seed 42 --plot-eval
python test.py --exp-name TEST/LAM_finetuned_Synth_dataset --eval-iter 1000000 --print-iter 0 --load-model /storage/brno2/home/xbabek02/LAM_finetuned.pth --dataset-name /storage/brno2/home/xbabek02/KNN --seed 42 --plot-eval
python test.py --exp-name TEST/Synth_pretrained_IAM_dataset --eval-iter 1000000 --print-iter 0 --load-model /storage/brno2/home/xbabek02/Synth_pretrained.pth --dataset-name iam --seed 42 --plot-eval
python test.py --exp-name TEST/Synth_pretrained_LAM_dataset --eval-iter 1000000 --print-iter 0 --load-model /storage/brno2/home/xbabek02/Synth_pretrained.pth --dataset-name lam --seed 42 --plot-eval
python test.py --exp-name TEST/Synth_pretrained_Synth_dataset --eval-iter 1000000 --print-iter 0 --load-model /storage/brno2/home/xbabek02/Synth_pretrained.pth --dataset-name /storage/brno2/home/xbabek02/KNN --seed 42 --plot-eval
```

grafy loss:
```sh
python train.py --exp-name TRAIN/10x5k  --dataset-name /storage/brno2/home/xbabek02/KNN --epochs 10 --eval-iter 1000 --total-iter 5000    --train-bs 8
python train.py --exp-name TRAIN/5x10k  --dataset-name /storage/brno2/home/xbabek02/KNN --epochs 5  --eval-iter 1000 --total-iter 10000   --train-bs 8
python train.py --exp-name TRAIN/2x25k  --dataset-name /storage/brno2/home/xbabek02/KNN --epochs 2  --eval-iter 1000 --total-iter 25000   --train-bs 8
python train.py --exp-name TRAIN/1x50k  --dataset-name /storage/brno2/home/xbabek02/KNN --epochs 1  --eval-iter 1000 --total-iter 50000   --train-bs 8

python train.py --exp-name TRAIN\20x10k  --dataset-name /storage/brno2/home/xbabek02/KNN --epochs 100 --eval-iter 1000 --total-iter 5000   --train-bs 32
python train.py --exp-name TRAIN\2x100k  --dataset-name /storage/brno2/home/xbabek02/KNN --epochs 5   --eval-iter 1000 --total-iter 100000 --train-bs 32
python train.py --exp-name TRAIN\1x200k  --dataset-name /storage/brno2/home/xbabek02/KNN --epochs 1   --eval-iter 1000 --total-iter 200000 --train-bs 32

python train.py --exp-name TRAIN\gpt-it\100x5k  --dataset-name /storage/brno2/home/xbabek02/KNN --epochs 100 --eval-iter 1000 --total-iter 5000   --train-bs 32 --dec-model gpt2
python train.py --exp-name TRAIN\gpt-it\100x5k  --dataset-name /storage/brno2/home/xbabek02/KNN --epochs 100 --eval-iter 1000 --total-iter 5000   --train-bs 32 --dec-model gpt2-it
```

trénování:
```sh
python train.py ...  
```

finetuning:
```sh
python train.py --load-model path/to/model.pth ...
```


### grafy

trénování:
```sh
cd output/TRAIN
python train_proportions.py
```

testování:
```sh
py print.py --path ./output/TEST/LAM_finetuned_IAM_dataset/  
py print.py --path ./output/TEST... 
```