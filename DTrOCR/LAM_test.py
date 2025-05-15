
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
import random
from lam_dataset import LAM

import kagglehub

from typing import Literal
import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from dataset import SynthtigerDataset
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


def show_plots(cers: List[float], wers: List[float], preds_list: List[str], labels_list: List[str]):
    text_lengths = [len(l) for l in labels_list]
    
    stats = [cers, wers]
    shorts = ['CER', 'WER']
    labels = ['Character Error Rate (CER)', 'Word Error Rate (WER)']
    colors = ['dodgerblue', 'lightcoral']
    colormaps = ['Blues', 'Reds']
    
    sns.set_style("whitegrid")
    
    for stat, short, label, color, colormap in zip(stats, shorts, labels, colors, colormaps):
        # Histogram
        plt.figure(figsize=(8, 6))
        sns.histplot(stat, bins=50, kde=True, color=color)
        plt.title(f"Histogram of {label}")
        plt.ylabel("Frequency")
        plt.xlabel(short)
        plt.yscale('log')
        plt.ylim(bottom=.75)
        plt.grid(True)
        
        # plt.figure(figsize=(7, 6))
        # sns.scatterplot(x=stat, y=text_lengths, alpha=0.1, color=color)
        # plt.title(f"{label} vs. Text Length")
        # plt.ylabel("Text Length (Ground Truth)")
        # plt.xlabel(short)
        # plt.grid(True)
        
        # Create the plot
        plt.figure(figsize=(7, 6))
        sns.kdeplot(y=text_lengths, x=stat, fill=False, cmap=colormap, alpha=0.4, levels=20, thresh=0.02)
        sns.scatterplot(y=text_lengths, x=stat, alpha=0.2, label="Data Points", s=20, edgecolor='w', linewidth=0.3, color=color)

        # 3. Add titles and labels
        plt.title(f"{label} vs Text Length")
        plt.ylabel("Text Length (Ground Truth)")
        plt.xlabel(short)
        plt.grid(True, linestyle='--', alpha=0.7)

        if len(text_lengths) > 0 and len(stat) > 0:
            plt.ylim(bottom=max(-1, np.min(text_lengths) - 2), top=np.max(text_lengths) + 5)
            plt.xlim(left=max(-0.015, np.min(stat) - 0.1), right=np.max(stat) + 0.2)
            
        plt.legend()
        plt.tight_layout()

    utils.plot_most_mismatched_words(preds_list, labels_list, top_n=10)

    plt.show()

def store_predictions(predictions, gt, dst = 'predictions.pkl'):
    try:
        with open(dst, 'wb') as f:
            pickle.dump([predictions, gt], f)
        print(f"Lists stored to {dst}")
    except IOError as e:
        print(f"Error storing lists with pickle: {e}")

def load_predictions(src = 'predictions.pkl'):
    try:
        with open(src, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Pickle file '{src}' not found")
    except (pickle.UnpicklingError, EOFError) as e:
        print(f"Error loading lists with pickle: {e}")



def load_lam_dataloaders(config:DTrOCRConfig,
                         charset=None, 
                         batch_size=100, 
                         eval_mode=False, 
                         dataset_name: Literal["vpippi/lam-dataset", "local"] = "vpippi/lam-dataset", 
                         subset=None, 
                         seed=42):
    match dataset_name:
        case "vpippi/lam-dataset":
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
                
        case "local":
            path = "datasets_f/lam-dataset"
            print("Path to dataset files:", path)

            gt_file = "gt_basic.txt"
            datasets_locations = ["../datasets_f/dataset1", "../datasets_f/dataset2"]

            # read samples
            samples = []
            for root_dir in datasets_locations:
                samples += load_samples_from_directory(root_dir, gt_file)

            random.seed(seed)
            random.shuffle(samples)

            full_data = SynthtigerDataset(config=config, samples=samples)

            n = len(samples)
            print(f"Number of samples = {n}")

            n_train = int(0.9 * n)
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
            
        case _:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
    
    if eval_mode:
        return test_loader
    else:
        return train_loader, val_loader, test_loader

def main_train():
    config = DTrOCRConfig(
        # attn_implementation='flash_attention_2'
        alphabet_size=len(alphabet)
    )

    args = option.get_args_parser()
    torch.manual_seed(args.seed)

    args.save_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(args.save_dir, exist_ok=True)

    logger = utils.get_logger(args.save_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    writer = SummaryWriter(args.save_dir)

    dev_str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(dev_str)

    model = DTrOCRLMHeadModel(config)

    total_param = sum(p.numel() for p in model.parameters())
    logger.info('total_param is {}'.format(total_param))

    model.train()
    model = model.to(device)
    model_ema = utils.ModelEma(model, args.ema_decay, device=device)
    model.zero_grad()

    optimizer = sam.SAM(model.parameters(), torch.optim.AdamW, lr=1e-7, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True)
    converter = utils.CTCLabelConverter(alphabet)

    train_dataloader, validation_dataloader, _ = load_lam_dataloaders(config=config, charset=alphabet, batch_size=75)

    best_cer, best_wer = 1e+6, 1e+6
    train_loss = 0.0
    step = 0
    EPOCHS = args.epochs

    checkpoint_path = os.path.join(args.save_dir, 'best_CER.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 3. Load states
    model.load_state_dict(checkpoint['model'])
    model_ema.ema.load_state_dict(checkpoint['state_dict_ema'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    for epoch in range(EPOCHS):
        logger.info(f'\nEpoch {epoch + 1}/{EPOCHS}')
        
        for step, inputs in enumerate(train_dataloader):
            label = inputs['label']

            del(inputs['label'])
            optimizer, current_lr = utils.update_lr_cos(step, args.warm_up_iter, args.total_iter, args.max_lr, optimizer)

            optimizer.zero_grad()
            text, length = converter.encode(label)
            batch_size = length.size(0)
            
            loss = compute_loss(args, model, inputs, batch_size, criterion, text, length, device)
            loss.backward()
            optimizer.first_step(zero_grad=True)
            compute_loss(args, model, inputs, batch_size, criterion, text, length, device).backward()
            optimizer.second_step(zero_grad=True)
            model.zero_grad()
            model_ema.update(model, num_updates=step / 2)
            train_loss += loss.item()

            if step % args.print_iter == 0:
                train_loss_avg = train_loss / args.print_iter

                logger.info(f'Iter : {step} \t LR : {current_lr:0.5f} \t training loss : {train_loss_avg:0.5f} \t ' )

                writer.add_scalar('./Train/lr', current_lr, step)
                writer.add_scalar('./Train/train_loss', train_loss_avg, step)
                train_loss = 0.0

            if step % args.eval_iter == 0:
                model.eval()
                with torch.no_grad():
                    val_loss, val_cer, val_wer, *_ = valid.validation(model_ema.ema,
                                                                                criterion,
                                                                                validation_dataloader,
                                                                                converter, device)

                    if val_cer < best_cer:
                        logger.info(f'CER improved from {best_cer:.4f} to {val_cer:.4f}!!!')
                        best_cer = val_cer
                        checkpoint = {
                            'model': model.state_dict(),
                            'state_dict_ema': model_ema.ema.state_dict(),
                            'optimizer': optimizer.state_dict(),
                        }
                        torch.save(checkpoint, os.path.join(args.save_dir, 'best_CER.pth'))

                    if val_wer < best_wer:
                        logger.info(f'WER improved from {best_wer:.4f} to {val_wer:.4f}!!!')
                        best_wer = val_wer
                        checkpoint = {
                            'model': model.state_dict(),
                            'state_dict_ema': model_ema.ema.state_dict(),
                            'optimizer': optimizer.state_dict(),
                        }
                        torch.save(checkpoint, os.path.join(args.save_dir, 'best_WER.pth'))

                    logger.info(
                        f'Val. loss : {val_loss:0.3f} \t CER : {val_cer:0.4f} \t WER : {val_wer:0.4f} \t ')

                    writer.add_scalar('./VAL/CER', val_cer, step)
                    writer.add_scalar('./VAL/WER', val_wer, step)
                    writer.add_scalar('./VAL/bestCER', best_cer, step)
                    writer.add_scalar('./VAL/bestWER', best_wer, step)
                    writer.add_scalar('./VAL/val_loss', val_loss, step)
                    model.train()



def main_evaluate():
    config = DTrOCRConfig(alphabet_size=len(alphabet))

    args = option.get_args_parser()
    torch.manual_seed(args.seed)

    args.save_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(args.save_dir, exist_ok=True)

    logger = utils.get_logger(args.save_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    writer = SummaryWriter(args.save_dir)

    dev_str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(dev_str)
    logger.info(f"Device: {device}")

    #
    try:
        test_dataloader = load_lam_dataloaders(config, alphabet, args.eval_bs, True, args.dataset_name, args.eval_iter, args.seed)
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
        logger.info(f"Overriding model path. Using specified path: {model_path}")

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
    total_param = sum(p.numel() for p in model_for_evaluation.parameters())
    trainable_param = sum(p.numel() for p in model_for_evaluation.parameters() if p.requires_grad)
    model_for_evaluation.eval()
    logger.info(f'Trainable params: {trainable_param}\n')
    

    # total = []
    logger.error(f"Evaluation of {eval_iters} batches, each batch having: {args.eval_bs} sample(s)\n")
    with torch.no_grad():
        # for batch in test_dataloader:
        test_loss, test_cer, test_wer, cers, wers, preds_list, labels_list = valid.validation(
            model_for_evaluation,
            criterion,
            test_dataloader,
            converter,
            device
        )
    
    # save predictions to a file
    store_predictions(preds_list, labels_list, os.path.join(args.save_dir, "predictions.pkl"))
    
    logger.info(f"Total Loss:  {test_loss:.4f}")
    logger.info(f"Total CER:   {test_cer:.4f}")
    logger.info(f"Total WER:   {test_wer:.4f}")
    print()
    logger.info(f"Average CER: {np.mean(cers):.4f}")
    logger.info(f"Median CER:  {np.median(cers):.4f}")
    logger.info(f"Std Dev CER: {np.std(cers):.4f}")
    print()
    logger.info(f"Average WER: {np.mean(wers):.4f}")
    logger.info(f"Median WER:  {np.median(wers):.4f}")
    logger.info(f"Std Dev WER: {np.std(wers):.4f}")
    
    writer.add_scalar('./Test/loss', test_loss, 0)
    writer.add_scalar('./Test/cer', test_cer, 0)
    writer.add_scalar('./Test/wer', test_wer, 0)    
    
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
        show_plots(cers, wers, preds_list, labels_list)
    
    

if __name__ == '__main__':
    main_evaluate()
    # preds_list, labels_list = load_predictions(os.path.join("output", "test", "predictions.pkl"))
    # utils.plot_most_mismatched_words(preds_list, labels_list, 15)
    # plt.show()

