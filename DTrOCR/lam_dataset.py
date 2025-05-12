import os
import json
import torch
from processor import DTrOCRProcessor
from config import DTrOCRConfig
from PIL import Image

class LAM(torch.utils.data.Dataset):
    def __init__(self,
                 db_path,
                 split,
                 nameset='train',
                 charset=None,
                 config=None,
                 max_target_length=64):
        set_path = os.path.join(db_path, 'lines', 'split', split, f'{nameset}.json')
        assert os.path.exists(set_path)

        with open(set_path, 'r') as f:
            self.samples = json.load(f)
        self.db_path = db_path
        self.imgs_path = os.path.join(db_path, 'lines', 'img')
        self.processor = DTrOCRProcessor(config, add_eos_token=True, add_bos_token=True)
        self.max_target_length = max_target_length

        if charset is None:
            labels = [sample['text'] for sample in self.samples]
            charset = sorted(set(''.join(labels)))
        self.charset = charset

        # Filter out any characters not in the allowed charset
        for sample in self.samples:
            filtered_text = ''.join([c for c in sample['text'] if c in self.charset])
            sample['text'] = filtered_text
        
        self.char_to_idx = dict(zip(self.charset, range(len(self.charset))))
        self.idx_to_char = dict(zip(range(len(self.charset)), self.charset))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        img_name, text, decade_id = sample['img'], sample['text'], sample['decade_id']

        img = Image.open(os.path.join(self.imgs_path, img_name)).convert("RGB")

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


