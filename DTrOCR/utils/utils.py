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
import matplotlib.pyplot as plt
import glob
import re
import numpy as np
    

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
import re
from typing import List, Tuple


import matplotlib.pyplot as plt


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


@dataclass
class Word:
    id: str
    file_path: Path
    writer_id: str
    transcription: str

def get_words_from_xml(xml_file, word_image_files):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    root_id = root.get('id')
    writer_id = root.get('writer-id')
    xml_words = []
    for line in root.findall('handwritten-part')[0].findall('line'):
        for word in line.findall('word'):
            image_file = Path([f for f in word_image_files if f.endswith(word.get('id') + '.png')][0])
            try:
                with Image.open(image_file) as _:
                    xml_words.append(
                        Word(
                            id=root_id,
                            file_path=image_file,
                            writer_id=writer_id,
                            transcription=word.get('text')
                        )
                    )
            except Exception:
                print(f"Error opening image file: {image_file}")
            
    return xml_words

    

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




def show_image(image, transcription, predicted_text):
    plt.figure()
    plt.title(f"True: {transcription}    Predicted: {predicted_text}", fontsize=20)
    plt.imshow(np.array(image, dtype=np.uint8))
    plt.xticks([]), plt.yticks([])
    plt.show()

def prediction_histogram(word_histogram):
    # count word frequencies
    true_counts = Counter(word_histogram["true"])
    predicted_counts = Counter(word_histogram["predicted"])
    all_words = list(set(true_counts.keys()).union(predicted_counts.keys()))

    # prepare data for plotting
    true_freqs = [true_counts[word] for word in all_words]
    predicted_freqs = [predicted_counts[word] for word in all_words]

    x = range(len(all_words))
    width = 0.4

    plt.figure(figsize=(10, 6))
    plt.bar([i - width/2 for i in x], true_freqs, width=width, label='Ground Truth')
    plt.bar([i + width/2 for i in x], predicted_freqs, width=width, label='Predicted')
    plt.xticks(x, all_words, rotation=90)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Word Frequency Histogram: Ground Truth vs. Predicted')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_train_val_loss():
    with open("LAM_train_val.pkl", 'rb') as f:
        data = pickle.load(f)
        
    
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # calculate epoch steps
        # epoch_interval = 2478.75 # TODO
        # max_step = data['Step'].max()
        # epoch_steps = [step for step in range(0, int(max_step + epoch_interval), int(epoch_interval))]
        
        x = range(1, len(data['train_losses']) + 1)

        # Plot Losses
        ax1.plot(x, data['train_losses'], label='Training Loss', marker='o')
        ax1.plot(x, data['validation_losses'], label='Validation Loss', marker='s')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')

        # Second y-axis for CER
        # ax2 = ax1.twinx()
        # ax2.plot(data['Step'], data['Cer'], label='CER', color='gray', linestyle='--', marker='^', alpha=0.6)
        # ax2.set_ylabel('CER')

        # Add vertical lines for epochs
        # for i, epoch_step in enumerate(epoch_steps):
        #     if i == 0:
        #         continue
        #     ax1.axvline(x=epoch_step, color='black', linestyle='--', linewidth=1, label='Epoch Marker' if i == 1 else None)


        # Combine legends
        lines, labels = ax1.get_legend_handles_labels()
        # lines2, labels2 = ax2.get_legend_handles_labels()
        # ax1.legend(lines + lines2, labels + labels2, loc='upper right')
        ax1.legend(lines, labels, loc='upper right')

        plt.title("Loss and CER over Steps with Epoch Markers")
        plt.grid(True)
        plt.tight_layout()
        plt.show()



def plot_loss_acc():
    files = sorted(glob.glob("LAM_epoch_*.pkl"), key=lambda x: int(re.findall(r'\d+', x)[0]))

    losses = []
    accuracies = []

    for file in files:
        with open(file, 'rb') as f:
            data = pickle.load(f)
            # losses.append(sum(data['loss']) / len(data['loss']))
            # accuracies.append(sum(data['acc']) / len(data['acc']))
            losses.extend(data['loss'])
            accuracies.extend(data['acc'])

    # steps & epochs
    step_interval = 5
    steps = [i * step_interval for i in range(len(losses))]
    epoch_interval = 1250 * 5
    epochs = [i * epoch_interval for i in range(len(files))]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    # losses
    ax1.plot(steps, losses, label='Training Loss', alpha=0.6)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    # acc
    ax2 = ax1.twinx()
    ax2.plot(steps, accuracies, label='Accuracy', color='green', linestyle='--', alpha=0.6)
    ax2.set_ylabel('Accuracy')

    # vertical lines at each epoch
    for i, epoch in enumerate(epochs):
        if i == 0:
            continue
        ax1.axvline(x=epoch, color='black', linestyle='--', linewidth=1, label='Epoch Marker' if i == 1 else None)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.title("Training Loss and Accuracy over Steps")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    

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

def plot_most_mismatched_words(
    predict_labels: List[str],
    gt_labels: List[str],
    top_n: int = 20,
    figsize: Tuple[int, int] = (8, 6)
):
    mismatch_counts = get_mismatched_word_pairs(predict_labels, gt_labels)

    # Prepare data for plotting
    most_common_mismatches = mismatch_counts.most_common(top_n)
    labels = [f"'{gt}' → '{pred}'" for (gt, pred), _ in most_common_mismatches]
    counts = [count for _, count in most_common_mismatches]

    if not labels:
        print("No mismatch data to plot after filtering.")
        return

    
    # Create a bar plot
    plt.figure(figsize=figsize)
    df_plot = pd.DataFrame({'Mismatch': labels, 'Frequency': counts})
    bar_plot = sns.barplot(x='Frequency', y='Mismatch', data=df_plot, hue='Mismatch', palette="viridis", legend=False)
    plt.title(f"Top {min(top_n, len(labels))} Most Frequent Word Mismatches")
    plt.xlabel("Frequency of Mismatch")
    plt.ylabel("Mismatched Pair (Ground Truth → Prediction)")
    
    for i, v in enumerate(counts):
        bar_plot.text(v + (max(counts) * 0.01), i, str(v), color='black', va='center', fontweight='medium')
        
    plt.tight_layout()
