import torch
import torch.distributed as dist
from torch.distributions.uniform import Uniform

import os
import re
import sys
import math
import logging
from copy import deepcopy
from collections import OrderedDict



import matplotlib.pyplot as plt
from collections import Counter

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image

import Levenshtein
import re

import pickle
import re
import numpy as np
    

import seaborn as sns
import pandas as pd
from collections import Counter
import re
from typing import List, Tuple


import matplotlib.pyplot as plt
import editdistance


from sklearn.metrics import accuracy_score, f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def randint(low, high):
    return int(torch.randint(low, high, (1, )))


def rand_uniform(low, high):
    return float(Uniform(low, high).sample())


def get_logger(out_dir):
    logger = logging.getLogger('Exp')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_path = os.path.join(out_dir, "run.log")
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)
    return logger


def update_lr_cos(nb_iter, warm_up_iter, total_iter, max_lr, optimizer, min_lr=1e-7):

    if nb_iter < warm_up_iter:
        current_lr = max_lr * (nb_iter + 1) / (warm_up_iter + 1)
    else:
        current_lr = min_lr + (max_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * nb_iter / (total_iter - warm_up_iter)))

    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr


class CTCLabelConverter(object):
    def __init__(self, characters):
        dict_chars = {char: idx for idx, char in enumerate(characters)}
        dict_character = list(dict_chars)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i + 1
        if len(self.dict) == 87:     # '[' and ']' are not in the test set but in the training and validation sets.
            self.dict['['], self.dict[']'] = 88, 89
        self.character = ['[blank]'] + dict_character

    def encode(self, text):
        length = [len(s) for s in text]
        text = ''.join(text)
        text = [self.dict[char] for char in text]

        return (torch.IntTensor(text).to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        texts = []
        index = 0

        for l in length:
            t = text_index[index:index + l]
            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])) and t[i]<len(self.character):
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
            index += l
        return texts


class Averager(object):
    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


class Metric(object):
    def __init__(self, name=''):
        self.name = name
        self.sum = torch.tensor(0.).double()
        self.n = torch.tensor(0.)

    def update(self, val):
        rt = val.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= dist.get_world_size()
        self.sum += rt.detach().cpu().double()
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n.double()


class ModelEma:
    def __init__(self, model, decay=0.9999, device='', resume=''):
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.device = device
        if device:
            self.ema.to(device=device)
        self.ema_has_module = hasattr(self.ema, 'module')
        if resume:
            self._load_checkpoint(resume)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def _load_checkpoint(self, checkpoint_path, mapl=None):
        checkpoint = torch.load(checkpoint_path,map_location=mapl)
        assert isinstance(checkpoint, dict)
        if 'state_dict_ema' in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict_ema'].items():
                if self.ema_has_module:
                    name = 'module.' + k if not k.startswith('module') else k
                else:
                    name = k
                new_state_dict[name] = v
            self.ema.load_state_dict(new_state_dict)
            print("=> Loaded state_dict_ema")
        else:
            print("=> Failed to find state_dict_ema, starting from loaded model weights")

    def update(self, model, num_updates=-1):
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        if num_updates >= 0:
            _cdecay = min(self.decay, (1 + num_updates) / (10 + num_updates))
        else:
            _cdecay = self.decay

        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema.state_dict().items():
                if needs_module:
                    k = 'module.' + k
                model_v = msd[k].detach()
                if self.device:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(ema_v * _cdecay + (1. - _cdecay) * model_v)



### cer wer operations

def format_string_for_wer(s: str) -> str:
    s = re.sub(r'([\[\]{}/\\()\"\'&+*=<>?.;:,!\-—_€#%°])', r' \1 ', s)
    s = re.sub(r'([ \n])+', " ", s).strip()
    return s

# Normalizes the text by converting to lowercase, removing punctuation, and collapsing multiple whitespaces.
def normalize_text(text: str) -> str:
    text = re.sub(r'[^\w\s]', '', text.lower())
    text = ' '.join(text.split())
    return text

    
def _normalize_word(word: str) -> str:
    word = word.lower()
    word = re.sub(r"[.,!?;:\"\'()`\[\]{}]", "", word)
    return word.strip()

def get_acc_f1(labels_list, preds_list):
    accuracy = accuracy_score(labels_list, preds_list)
    f1 = f1_score(labels_list, preds_list, average='macro')
    return accuracy, f1

def calc_total_wer_cer(labels_list, preds_list):
    norm_ED = 0
    norm_ED_wer = 0

    tot_ED = 0
    tot_ED_wer = 0

    length_of_gt = 0
    length_of_gt_wer = 0

    for pred_cer, gt_cer in zip(preds_list, labels_list):
        tmp_ED = editdistance.eval(pred_cer, gt_cer)
        if len(gt_cer) == 0:
            norm_ED += 1
        else:
            norm_ED += tmp_ED / float(len(gt_cer))
        tot_ED += tmp_ED
        length_of_gt += len(gt_cer)

    for pred_wer, gt_wer in zip(preds_list, labels_list):
        pred_wer = format_string_for_wer(pred_wer)
        gt_wer = format_string_for_wer(gt_wer)
        pred_wer = pred_wer.split()
        gt_wer = gt_wer.split()
        tmp_ED_wer = editdistance.eval(pred_wer, gt_wer)

        if len(gt_wer) == 0:
            norm_ED_wer += 1
        else:
            norm_ED_wer += tmp_ED_wer / float(len(gt_wer))

        tot_ED_wer += tmp_ED_wer
        length_of_gt_wer += len(gt_wer)

    CER = tot_ED / float(length_of_gt)
    WER = tot_ED_wer / float(length_of_gt_wer)
    
    return CER, WER

    

def calculate_cer(ground_truth: str, prediction: str, normalize: bool = False) -> float:
    if normalize:
        ground_truth = normalize_text(ground_truth)
        prediction = normalize_text(prediction)

    if not ground_truth:
        return 0.0 if not prediction else 1.0

    # number of edits
    distance = Levenshtein.distance(prediction, ground_truth)

    return distance / len(ground_truth)

def calculate_wer(ground_truth: str, prediction: str, normalize: bool = False) -> float:
    if normalize:
        ground_truth = normalize_text(ground_truth)
        prediction = normalize_text(prediction)

    # split strings into words (based on whitespace)
    ground_truth_words = ground_truth.split()
    prediction_words = prediction.split()

    if not ground_truth_words:
        return 0.0 if not prediction_words else 1.0

    # calculate Levenshtein distance at the word level
    distance = Levenshtein.distance(prediction_words, ground_truth_words)

    return distance / len(ground_truth_words)

def calculate_cer_wer(ground_truth: str, prediction: str) -> tuple:
    # calculate WITHOUT normalization
    cer_raw = calculate_cer(ground_truth, prediction)
    wer_raw = calculate_wer(ground_truth, prediction)

    # calculate WITH normalization
    norm_gt = normalize_text(ground_truth)
    norm_pred = normalize_text(prediction)

    cer_norm = calculate_cer(norm_gt, norm_pred)
    wer_norm = calculate_wer(norm_gt, norm_pred)

    print(f"GROUND TRUTH:    '{ground_truth}'")
    print(f"PREDICTION:      '{prediction}'")
    print(f"GT normalized:   '{norm_gt}'")
    print(f"PRED normalized: '{norm_pred}'")
    print()

    # CER & WER
    print(f"Raw CER: {cer_raw:.4f} ({cer_raw*100:.2f}%)")
    print(f"Raw WER: {wer_raw:.4f} ({wer_raw*100:.2f}%)")
    print()

    print(f"Normalized CER: {cer_norm:.4f} ({cer_norm*100:.2f}%)")
    print(f"Normalized WER: {wer_norm:.4f} ({wer_norm*100:.2f}%)")
    print()

    # Accuracy
    print(f"Normalized Character Accuracy: {(1 - cer_norm):.4f} ({(1 - cer_norm)*100:.2f}%)")
    print(f"Normalized Word Accuracy:      {(1 - wer_norm):.4f} ({(1 - wer_norm)*100:.2f}%)")
    print(f"\n{'='*50}\n")

    return (cer_raw, wer_raw), (cer_norm, wer_norm)
    

def get_mismatched_word_pairs(
    predict_labels: List[str],
    gt_labels: List[str]
) -> Counter:
    mismatch_counts = Counter()

    for pred_sentence, gt_sentence in zip(predict_labels, gt_labels):
        pred_words_raw = pred_sentence.split()
        gt_words_raw = gt_sentence.split()

        # Normalize words
        pred_words = [_normalize_word(w) for w in pred_words_raw if _normalize_word(w)] # Filter empty after norm
        gt_words = [_normalize_word(w) for w in gt_words_raw if _normalize_word(w)]   # Filter empty after norm
        len_to_compare = min(len(pred_words), len(gt_words))

        for i in range(len_to_compare):
            pred_word = pred_words[i]
            gt_word = gt_words[i]

            if pred_word != gt_word:
                mismatch_counts[(gt_word, pred_word)] += 1
                
    return mismatch_counts



### load store resulting arrays

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


