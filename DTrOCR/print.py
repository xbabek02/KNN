import os
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from wordcloud import WordCloud

from sklearn.metrics import confusion_matrix

from collections import Counter, defaultdict

from utils import utils 

import Levenshtein
import glob
import pickle

import argparse
import re # Added

_DEFAULT_INITIAL_PATH = "output"
os.makedirs(_DEFAULT_INITIAL_PATH, exist_ok=True)
pth = lambda name: os.path.join(_DEFAULT_INITIAL_PATH, name)

def path(path="output"):
    os.makedirs(path, exist_ok=True)
    return lambda name: os.path.join(path, name)

def get_args_parser():
    parser = argparse.ArgumentParser(description='Print evaluation results', add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', type=str, default='output', help='Select path to a files')
    
    return parser.parse_args()



### training

def plot_train_val_loss():
    with open(pth("LAM_train_val.pkl"), 'rb') as f:
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
    files = sorted(glob.glob(pth("LAM_epoch_*.pkl")), key=lambda x: int(re.findall(r'\d+', x)[0]))

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







### evaluation / testing


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


def plot_most_mismatched_words(
    predict_labels: list[str],
    gt_labels: list[str],
    top_n: int = 10,
    figsize: tuple[int, int] = (8, 6)
):
    mismatch_counts = utils.get_mismatched_word_pairs(predict_labels, gt_labels)

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
    plt.savefig(pth("mismatched_words.png"), dpi=600, bbox_inches='tight')




def plot_position_index(preds_list, labels_list):
     # Character Position Error
    char_position_errors = defaultdict(int)
    char_position_counts = defaultdict(int)

    for pred, label in zip(preds_list, labels_list):
        min_len = min(len(pred), len(label))
        for i in range(min_len):
            char_position_counts[i] += 1
            if pred[i] != label[i]:
                char_position_errors[i] += 1

    char_positions = sorted(char_position_errors.keys())
    char_error_rates = [
        char_position_errors[pos] / char_position_counts[pos]
        for pos in char_positions
    ]

    # Word Position Error
    word_position_errors = defaultdict(int)
    word_position_counts = defaultdict(int)

    for pred, label in zip(preds_list, labels_list):
        pred_words = pred.split()
        label_words = label.split()
        min_len = min(len(pred_words), len(label_words))
        for i in range(min_len):
            word_position_counts[i] += 1
            if pred_words[i] != label_words[i]:
                word_position_errors[i] += 1

    word_positions = sorted(word_position_errors.keys())
    word_error_rates = [
        word_position_errors[pos] / word_position_counts[pos]
        for pos in word_positions
    ]

    plt.figure(figsize=(7, 4))
    plt.plot(char_positions, char_error_rates, marker='o', label='CER by Char Position', color='dodgerblue')
    plt.plot(word_positions, word_error_rates, marker='s', label='WER by Word Position', color='lightcoral')
    plt.title("Error Rate by Character and Word Position")
    plt.xlabel("Position Index")
    plt.ylabel("Error Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(pth(f"position_index.png"), dpi=600, bbox_inches='tight')



def word_cloud(preds_list, labels_list):
    mistakes = []
    for pred, label in zip(preds_list, labels_list):
        pred_words = pred.split()
        label_words = label.split()
        for pw, lw in zip(pred_words, label_words):
            if pw != lw:
                mistakes.append(lw)

    wc = WordCloud(width=800, height=400, background_color='white').generate(" ".join(mistakes))
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title("Most Frequently Misrecognized Words")
    plt.tight_layout()
    plt.savefig(pth(f"word_cloud.png"), dpi=600, bbox_inches='tight')


def char_confusion_matrix(labels_list, preds_list):
    true_chars = []
    pred_chars = []

    for true_str, pred_str in zip(labels_list, preds_list):
        min_len = min(len(true_str), len(pred_str))
        true_chars.extend(list(true_str[:min_len]))
        pred_chars.extend(list(pred_str[:min_len]))

    # Character set
    char_set = sorted(set(true_chars + pred_chars))

    # Confusion matrix
    cm = confusion_matrix(true_chars, pred_chars, labels=char_set)
    cm_safe = np.where(cm == 0, 1e-6, cm)

    fig, ax = plt.subplots(figsize=(12, 10))
    cax = ax.imshow(cm_safe, interpolation='nearest', cmap='Blues',
                    norm=LogNorm(vmin=cm_safe.min(), vmax=cm_safe.max()))
    ax.set_xticks(np.arange(len(char_set)))
    ax.set_yticks(np.arange(len(char_set)))
    ax.set_xticklabels(char_set, fontsize=8)
    ax.set_yticklabels(char_set, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title("Character-level Confusion Matrix (Log Scale)", fontsize=12)
    fig.colorbar(cax)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(pth(f"char_conf_matrix.png"), dpi=600, bbox_inches='tight')
    
    
def heatmap(data):
    # Create bins for text lengths
    m = np.arange(0, max(data['TextLength'])+1)

    # Pivot table for heatmap
    heatmap_data = data.pivot_table(index='LengthBin', values=['CER', 'WER'], aggfunc='mean', observed=False)

    plt.figure(figsize=(16, 3))
    sns.heatmap(heatmap_data.T, annot=True, fmt=".2f", cmap='coolwarm', annot_kws={"size": 6})
    plt.title('Average CER/WER by Text Length Bins')
    plt.ylabel('Error Type')
    plt.xlabel('Text Length Bins')
    plt.xticks([i-0.5 for i in m if i > 0], [str(i) for i in m if i > 0], rotation=0)
    plt.tight_layout()
    plt.savefig(pth(f"heatmap.png"), dpi=600, bbox_inches='tight')
    
    

def comparisons(cers, wers, text_lengths):
    stats = [cers, wers]
    shorts = ['CER', 'WER']
    labels = ['Character Error Rate (CER)', 'Word Error Rate (WER)']
    colors = ['dodgerblue', 'lightcoral']
    colormaps = ['Blues', 'Reds']
    
    sns.set_style("whitegrid")
    
    # Histogram both together
    plt.figure(figsize=(6, 4))
    for stat, short, label, color, colormap in zip(stats, shorts, labels, colors, colormaps):
        sns.histplot(stat, bins=50, kde=True, color=color, label=short)
    plt.title(f"Histogram of WER and CER")
    plt.ylabel("Frequency")
    plt.xlabel(short)
    plt.yscale('log')
    plt.ylim(bottom=.75)
    plt.grid(True)
    plt.legend()
    plt.savefig(pth(f"histogram_WER_CER.png"), dpi=600, bbox_inches='tight')
    
    
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
        plt.savefig(pth(f"histogram_{short}.png"), dpi=600, bbox_inches='tight')
        
        # Create the plot
        plt.figure(figsize=(7, 6))
        sns.kdeplot(y=text_lengths, x=stat, fill=False, cmap=colormap, alpha=0.6, levels=10, thresh=0.02)
        sns.scatterplot(y=text_lengths, x=stat, alpha=0.2, s=10, edgecolor='w', linewidth=0.3, color=color)

        plt.title(f"{label} vs Text Length")
        plt.ylabel("Text Length (Ground Truth)")
        plt.xlabel(short)
        plt.grid(True, linestyle='--', alpha=0.7)

        if len(text_lengths) > 0 and len(stat) > 0:
            plt.ylim(bottom=max(-1, np.min(text_lengths) - 2), top=np.max(text_lengths) + 5)
            plt.xlim(left=max(-0.015, np.min(stat) - 0.1), right=np.max(stat) + 0.2)
            
        plt.tight_layout()
        plt.savefig(pth(f"kde_scatter_{short}.png"), dpi=600, bbox_inches='tight')


def show_plots(cers: list[float], wers: list[float], preds_list: list[str], labels_list: list[str], save_dir = None):
    global pth
    if save_dir:
        pth = path(save_dir)
    text_lengths = [len(l) for l in labels_list]
    distances = [Levenshtein.distance(p, l) for p, l in zip(preds_list, labels_list)]     
    m = np.arange(0, max(text_lengths)+1)
    
    data = pd.DataFrame({
        'CER': cers,
        'WER': wers,
        'Prediction': preds_list,
        'Label': labels_list,
        'TextLength': text_lengths,
        'Distance': distances,
        'LengthBin': pd.cut(text_lengths, bins=m)
    })
    
    # printing
    print_stats(cers, wers, preds_list, labels_list)
    print_worst(data)
    
    # plotting
    sns.set_style("whitegrid")
    
    comparisons(cers, wers, text_lengths)
    heatmap(data)
    char_confusion_matrix(labels_list, preds_list)
    plot_position_index(preds_list, labels_list)
    word_cloud(preds_list, labels_list)
    plot_most_mismatched_words(preds_list, labels_list)
    
    plt.show()
    
    

def print_stats(cers, wers, preds_list, labels_list):
    test_cer, test_wer = utils.calc_total_wer_cer(labels_list, preds_list)
    acc, f1 = utils.get_acc_f1(labels_list, preds_list)
    
    print(f"\n\nEvaluation Stats:")
    print(f"Total CER:   {test_cer:.4f}")
    print(f"Total WER:   {test_wer:.4f}")
    print()
    print(f"Accuracy:    {acc:.4f}")
    print(f"F1:          {f1:.4f}")
    print()
    print(f"Average CER: {np.mean(cers):.4f}")
    print(f"Median CER:  {np.median(cers):.4f}")
    print(f"Std Dev CER: {np.std(cers):.4f}")
    print()
    print(f"Average WER: {np.mean(wers):.4f}")
    print(f"Median WER:  {np.median(wers):.4f}")
    print(f"Std Dev WER: {np.std(wers):.4f}") 
    
    
def print_worst(data):
    top_n = 10
    worst_cases = data.sort_values(by='Distance', ascending=False).head(top_n)

    print(f"\n\nTop {top_n} Worst Cases:")
    for _, row in worst_cases.iterrows():
        print(f"Prediction: {row['Prediction']}")
        print(f"Label     : {row['Label']}")
        print(f"CER: {row['CER']:.2f}, WER: {row['WER']:.2f}, Distance: {row['Distance']}\n")




if __name__ == '__main__':
    args = get_args_parser()
    pth = path(args.path)
    preds_list, labels_list = utils.load_predictions(pth("predictions.pkl"))
    cers, wers = utils.load_predictions(pth("cers_wers.pkl"))
    
    show_plots(cers, wers, preds_list, labels_list)

