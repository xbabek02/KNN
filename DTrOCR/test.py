
import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pandas as pd

import os
import json
import valid
from utils import utils
from utils import sam
from utils import option
from model import DTrOCRLMHeadModel
from torch.utils.data import DataLoader

from config import DTrOCRConfig
from typing import Tuple
import tqdm
import glob
from pathlib import Path
import random
from lam_dataset import LAM

import kagglehub

from typing import Literal
import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from dataset import SynthtigerDataset, IAMDataset, Word, get_words_from_xml, get_all_words
from torch.utils.data import random_split
from torch.utils.data import Subset

import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List
from Levenshtein import distance as levenshtein_distance
import editdistance
import pickle

from KNN.DTrOCR.print import show_plots


import torch
from model import DTrOCRLMHeadModel
from typing import Tuple
import torch._dynamo
torch._dynamo.config.suppress_errors = True


alphabet = "!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~ ÈàèéìòùÀÈÉÌÒÙ’%"


def load_samples_from_directory(root_dir, gt_file):
    samples = []

    gt_path = os.path.join(root_dir, gt_file)
    if not os.path.exists(gt_path):
        print(f"Warning: {gt_path} not found. Skipping.")
        return
    
    with open(gt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                img_path, text = parts
                full_img_path = os.path.join(root_dir, img_path)
                samples.append((full_img_path, text))

    return samples


def normalize_for_display(tensor_img):
    min_val = tensor_img.min()
    max_val = tensor_img.max()
    normalized = (tensor_img - min_val) / (max_val - min_val + 1e-5)
    return normalized.permute(1, 2, 0).numpy()


def compute_loss(args, model, inputs, batch_size, criterion, text, length, pred_device):
    def send_inputs_to_device(dictionary, device):
        return {key: value.to(device=device) if isinstance(value, torch.Tensor) else value for key, value in dictionary.items()}

    inputs = send_inputs_to_device(inputs, pred_device)
    preds = model(**inputs)
    preds = preds.float()
    preds_size = torch.IntTensor([preds.size(1)] * batch_size).to(pred_device)
    preds = preds.permute(1, 0, 2).log_softmax(2)

    loss = criterion(preds.to(pred_device), text.to(pred_device), preds_size, length.to(pred_device)).mean()

    return loss

def compute_eval(model, model_ema, criterion, optimizer, validation_dataloader, converter, device, logger, writer, epoch, step, best_cer, best_wer, args, save=False):
    model.eval()
    with torch.no_grad():
        val_loss, val_cer, val_wer, *_ = valid.validation(model_ema.ema, criterion, validation_dataloader, converter, device, save, epoch, step)
        if val_cer < best_cer or save:
            logger.info(f'CER improved from {best_cer:.4f} to {val_cer:.4f}')
            best_cer = float(val_cer)
            checkpoint = {
                'model': model.state_dict(),
                'state_dict_ema': model_ema.ema.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(args.save_dir, f'best_CER{"_final" if save else ""}.pth'))

        if val_wer < best_wer or save:
            logger.info(f'WER improved from {best_wer:.4f} to {val_wer:.4f}')
            best_wer = float(val_wer)
            checkpoint = {
                'model': model.state_dict(),
                'state_dict_ema': model_ema.ema.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(args.save_dir, f'best_WER{"_final" if save else ""}.pth'))

        logger.info(f'Val. loss : {val_loss:0.3f} \t CER : {val_cer:0.4f} \t WER : {val_wer:0.4f}')

        writer.add_scalar('./VAL/CER', val_cer, step)
        writer.add_scalar('./VAL/WER', val_wer, step)
        writer.add_scalar('./VAL/bestCER', best_cer, step)
        writer.add_scalar('./VAL/bestWER', best_wer, step)
        writer.add_scalar('./VAL/val_loss', val_loss, step)
    model.train()
    
    return best_cer, best_wer

def load_dataloaders(config:DTrOCRConfig,
                    charset=None, 
                    batch_size=100, 
                    eval_mode=False, 
                    dataset_name: Literal["lam", "iam", "synth"] = "vpippi/lam-dataset", 
                    subset=None, 
                    seed=42):
    match dataset_name.lower():
        case "lam" | "vpippi/lam-dataset":
            dataset_name = "vpippi/lam-dataset"
            path = kagglehub.dataset_download(dataset_name)
            print("Path to dataset files:", path)
                    
            path_to_train_json = f"{path}/LAM/lines/split/basic/train.json"
            path_to_test_json = f"{path}/LAM/lines/split/basic/test.json"
            path_to_valid_json = f"{path}/LAM/lines/split/basic/val.json"
            path_to_images = f"{path}/LAM/lines/img/"

            lam_data_root = os.path.join(path, "LAM")

            assert os.path.exists(path_to_train_json) and os.path.exists(path_to_test_json) and os.path.exists(path_to_valid_json) and os.path.exists(path_to_images)
            
            train_dataset = LAM(lam_data_root, 'basic', charset=charset,  nameset='train', config=config)
            val_dataset = LAM(lam_data_root, 'basic', charset=charset,  nameset='val', config=config)
            test_dataset  = LAM(lam_data_root, 'basic', charset=charset,  nameset='test', config=config)
            
            if subset is not None:
                train_dataset = Subset(train_dataset, list(range(min(subset, len(train_dataset)))))
                val_dataset = Subset(val_dataset, list(range(min(subset, len(val_dataset)))))
                test_dataset = Subset(test_dataset, list(range(min(subset, len(test_dataset)))))

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
            test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        
        case "iam" | "iam_part":
            if os.path.exists(f'{dataset_name}/IAM_words.pkl'):
                words = pickle.load(open(f'{dataset_name}/IAM_words.pkl', 'rb'))
            else:
                words = get_all_words(dataset_name)
                pickle.dump(words, open(f'{dataset_name}/IAM_words.pkl', 'wb'))
            
            # train test validation split
            random.seed(seed)
            random.shuffle(words)
            train_split = int(0.8 * len(words))
            val_split = int(0.9 * len(words))
            train_word_records = words[:train_split]
            validation_word_records = words[train_split:val_split]
            test_word_records = words[val_split:]
            print(f"Train: {len(train_word_records)}, Validation: {len(validation_word_records)}, Test: {len(test_word_records)}")
            
            config = DTrOCRConfig()

            train_dataset = IAMDataset(words=train_word_records, config=config)
            val_dataset = IAMDataset(words=validation_word_records, config=config)
            test_dataset = IAMDataset(words=test_word_records, config=config)
            
            if subset is not None:
                train_dataset = Subset(train_dataset, list(range(min(subset, len(train_dataset)))))
                val_dataset = Subset(val_dataset, list(range(min(subset, len(val_dataset)))))
                test_dataset = Subset(test_dataset, list(range(min(subset, len(test_dataset)))))

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        case "synth" | "local" | _:
            gt_file = "gt_basic.txt"
            datasets_locations = [f"{dataset_name}/datasets_f/dataset1", f"{dataset_name}/datasets_f/dataset2"]
            
            # read samples
            samples = []
            for root_dir in datasets_locations:
                samples += load_samples_from_directory(root_dir, gt_file)
            
            random.seed(seed)
            random.shuffle(samples)
            
            full_data = SynthtigerDataset(config=config, samples=samples)
            
            n = len(samples)
            print(f"Number of samples = {n}")
            
            n_train = int(0.7 * n)
            n_valid = int(0.02 * n)
            n_test  = n - n_train - n_valid
            
            train_dataset, val_dataset, test_dataset = random_split(
                full_data,
                [n_train, n_valid, n_test],
                generator=torch.Generator().manual_seed(seed)
            )
            
            if subset is not None:
                train_dataset = Subset(train_dataset, list(range(min(subset, len(train_dataset)))))
                val_dataset = Subset(val_dataset, list(range(min(subset, len(val_dataset)))))
                test_dataset = Subset(test_dataset, list(range(min(subset, len(test_dataset)))))
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
    if eval_mode:
        return test_loader
    else:
        return train_loader, val_loader, test_loader


def main_evaluate():
    args = option.get_args_parser()
    torch.manual_seed(args.seed)

    args.save_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(args.save_dir, exist_ok=True)

    logger = utils.get_logger(args.save_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    dev_str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(dev_str)
    logger.info(f"Device: {device}")

    if args.dec_model == "gpt2":
        hf_model="openai-community/gpt2"
        vocab_size=50257
    elif args.dec_model == "gpt2-it":
        hf_model="LorenzoDeMattei/GePpeTto"
        vocab_size=30000
    
    config = DTrOCRConfig(
        gpt2_hf_model=hf_model,
        vocab_size=vocab_size,
        alphabet_size=len(alphabet)
    )
    
    #
    try:
        test_dataloader = load_dataloaders(config, alphabet, args.eval_bs, True, args.dataset_name, args.eval_iter, args.seed)
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading data: {e}", exc_info=True)
        return
    
    # print info about datasets
    eval_iters = len(test_dataloader.dataset)
    logger.info(f"Dataset: {args.dataset_name}")
    logger.info(f"Number of samples in test dataset: {eval_iters}\n")

    # 
    model_for_evaluation = DTrOCRLMHeadModel(config)
    criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True) 
    converter = utils.CTCLabelConverter(alphabet)

    #
    model_path = os.path.join(args.out_dir, 'LAM', 'LAM_finetuned.pth')
    if hasattr(args, 'load_model') and args.load_model:
        model_path = args.load_model
        logger.info(f"Using specified path: {model_path}")

    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return

    model = torch.load(model_path, map_location=device)
    logger.info(f"Loaded model {model_path}")

    # preferring EMA weights
    if 'state_dict_ema' in model:
        model_for_evaluation.load_state_dict(model['state_dict_ema'])
        logger.info("Loaded EMA model weights ('state_dict_ema') into the model.")
    elif 'model' in model:
        model_for_evaluation.load_state_dict(model['model'])
        logger.warning("EMA weights ('state_dict_ema') not found in model. Loaded standard model weights ('model').")
    else:
        logger.error("Checkpoint does not contain 'state_dict_ema' or 'model' keys. Cannot load model weights. Exiting.")
        return
    
    model_for_evaluation = model_for_evaluation.to(device)
    trainable_param = sum(p.numel() for p in model_for_evaluation.parameters() if p.requires_grad)
    model_for_evaluation.eval()
    logger.info(f'Trainable params: {trainable_param}\n')
    

    logger.info(f"Evaluation of {eval_iters} batches, each batch having: {args.eval_bs} sample(s)\n")
    with torch.no_grad():
        # for batch in test_dataloader:
        test_loss, test_cer, test_wer, cers, wers, preds_list, labels_list = valid.validation(
            model_for_evaluation,
            criterion,
            test_dataloader,
            converter,
            device,
            save=True
        )
    
    # save predictions to a file
    utils.store_predictions(preds_list, labels_list, os.path.join(args.save_dir, "predictions.pkl"))
    utils.store_predictions(cers, wers, os.path.join(args.save_dir, "cers_wers.pkl"))
    
    acc, f1 = utils.get_acc_f1(labels_list, preds_list)
    
    logger.info(f"Total Loss:  {test_loss:.4f}")
    logger.info(f"Total CER:   {test_cer:.4f}")
    logger.info(f"Total WER:   {test_wer:.4f}")
    print()
    logger.info(f"Accuracy:    {acc:.4f}")
    logger.info(f"F1:          {f1:.4f}")
    print()
    logger.info(f"Average CER: {np.mean(cers):.4f}")
    logger.info(f"Median CER:  {np.median(cers):.4f}")
    logger.info(f"Std Dev CER: {np.std(cers):.4f}")
    print()
    logger.info(f"Average WER: {np.mean(wers):.4f}")
    logger.info(f"Median WER:  {np.median(wers):.4f}")
    logger.info(f"Std Dev WER: {np.std(wers):.4f}") 
    
    for i, inputs in enumerate(test_dataloader):
        if i >= args.print_iter:
            break
        pred_text = preds_list[i]
        gt_text = labels_list[i]
        (cer_raw, wer_raw), (cer_norm, wer_norm) = utils.calculate_cer_wer(gt_text, pred_text)
        logger.info(f"Sample {i+1}")
        logger.info(f"  Predicted: '{pred_text}'")
        logger.info(f"  Actual:    '{gt_text}'")
        print()
        
        # imgs shape: (B, C, H, W)
        if args.plot_eval:
            for img in inputs['pixel_values']:
                image_np = normalize_for_display(img)
                plt.imshow(image_np)
                plt.title(f"Predicted:    {pred_text}\nGround truth: {gt_text}", loc='left')
                plt.axis('off')
                plt.show()
    
    if args.plot_eval:
        show_plots(cers, wers, preds_list, labels_list, args.save_dir)
    
if __name__ == '__main__':
    main_evaluate()
