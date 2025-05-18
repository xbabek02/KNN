import re
import matplotlib.pyplot as plt
import os
import numpy as np # For potential future smoothing

# calculates the moving average of a 1D array
def moving_average(data, window_size):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if len(data) < window_size:
        return np.array([])
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# calculates the moving standard deviation of a 1D array
def moving_std(data, window_size):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if len(data) < window_size:
        return np.array([])
    std_devs = []
    for i in range(len(data) - window_size + 1):
        window_data = data[i:i+window_size]
        std_devs.append(np.std(window_data))
    return np.array(std_devs)

def plot_simple_losses_with_ci(log_file_configs, window_size=5, txt=""):
    plt.figure(figsize=(7, 4))

    max_len = max([len(parse_simple_log(filepath))-1 for filepath, *_ in log_file_configs])

    for filepath, label, style in log_file_configs:
        losses = np.array(parse_simple_log(filepath)[1:])
        
        mean_smoothed = moving_average(losses, window_size)
        std_smoothed = moving_std(losses, window_size)

        sequential_steps = np.linspace(0, max_len, len(losses))
        x_smooth = sequential_steps[window_size-1:]
        x = np.linspace(0, max_len, len(x_smooth))
        
        std_smoothed = np.maximum(std_smoothed, 0)
        line, = plt.plot(x, mean_smoothed, label=label, linestyle=style)
        line_color = line.get_color()
        
        plt.fill_between(x, mean_smoothed - std_smoothed, mean_smoothed + std_smoothed,
                        color=line_color, alpha=0.2,
                        label=f'{label} (std dev band)' if len(log_file_configs) <= 2 else None)

    plt.xticks(np.linspace(0,max_len,11), [f"{int(x*100)}%" for x in np.linspace(0,1,11)])
    plt.xlabel("Traning proportion")
    plt.ylabel("Training Loss")
    plt.title(f"Training loss progression ({txt})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"losses_{txt}.png", dpi=600, bbox_inches='tight')

def parse_simple_log(filepath):
    losses_data = []
    # Regex to capture the training loss value
    log_pattern = re.compile(r"training loss\s*:\s*([\d.]+)")

    if not os.path.exists(filepath):
        print(f"Warning: File not found {filepath}")
        return [], []

    with open(filepath, 'r') as f:
        for line in f:
            match = log_pattern.search(line)
            if match:
                loss = float(match.group(1))
                losses_data.append(loss)
    return losses_data



if __name__ == '__main__':
    files_to_plot_config = [
        ('1x50_000.log', 'Run 1 (1x50k)', '-'),
        ('2x25_000.log', 'Run 2 (2x25k)', '--'),
        ('5x10_000.log', 'Run 3 (5x10k)', ':'),
        ('10x5_000.log', 'Run 4 (10x1k)', '-.') 
    ]
    
    plot_simple_losses_with_ci(files_to_plot_config, window_size=5, txt="50k total")
    
    files_to_plot_config = [
        ('1x200k.log', 'Run 1 (1x200k)', '-'),
        ('2x100k.log', 'Run 2 (2x100k)', '--'),
        ('20x10k.log', 'Run 3 (20x10k)', ':') 
    ]
    
    plot_simple_losses_with_ci(files_to_plot_config, window_size=5, txt="200k total")
    
    
    files_to_plot_config = [
        ('gpt2-orig.log', 'Run 1 (gpt2-orig)', '-'),
        ('gpt2-it.log', 'Run 2 (gpt2-it)', '--'),
    ]
    
    plot_simple_losses_with_ci(files_to_plot_config, window_size=15, txt="GPT2")
    
    plt.show()
