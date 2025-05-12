import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from processor import DTrOCRProcessor
from config import DTrOCRConfig

class SynthtigerDataset(Dataset):
    def __init__(self, config:DTrOCRConfig, samples:list):
        super(SynthtigerDataset, self).__init__()
        self.processor = DTrOCRProcessor(config, add_eos_token=True, add_bos_token=True)
        self.samples = samples


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except (FileNotFoundError, OSError) as e:
            print(f"Warning: Could not load image at {img_path}: {e}")
            
            dummy_img = Image.new('RGB', (224, 224), color='black')
            img = dummy_img
            text = " "

        inputs = self.processor(
            images=img,
            texts=text,
            padding='max_length',
            return_tensors="pt",
        )
        return {
            'pixel_values': inputs.pixel_values[0],
            'input_ids': inputs.input_ids[0],
            'attention_mask': inputs.attention_mask[0],
            'label':text
        }