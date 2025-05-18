import os
import torch
import tqdm
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import xml.etree.ElementTree as ET

from dataclasses import dataclass

from processor import DTrOCRProcessor
from config import DTrOCRConfig
from utils import utils

from functools import partial
import multiprocessing as mp

### LAM

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
        
        
        

### IAM
        
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

def get_words_from_xml_optimized(xml_file_path_str: str, image_path_map: dict):    
    xml_file_path = Path(xml_file_path_str)
    
    try:
        tree = ET.parse(xml_file_path)
    except ET.ParseError:
        print(f"Error parsing XML file: {xml_file_path}")
        return []
        
    root = tree.getroot()
    form_id = root.get('id')
    writer_id = root.get('writer-id')
    xml_words_data = []
    
    handwritten_part = root.find('handwritten-part')
    if handwritten_part is None:
        return xml_words_data

    for line in handwritten_part.findall('line'):
        for word_node in line.findall('word'):
            word_id = word_node.get('id') 
            transcription = word_node.get('text')
            image_filename_key = f"{word_id}.png"
            
            if image_filename_key in image_path_map:
                image_full_path = image_path_map[image_filename_key]
                try:
                    with Image.open(image_full_path) as img_test:
                        if img_test.format is None:
                            continue 

                    xml_words_data.append(
                        Word(
                            id=word_id,
                            file_path=image_full_path,
                            writer_id=writer_id,
                            transcription=transcription
                        )
                    )
                except FileNotFoundError:
                    print(f"Error: Image file {image_full_path} (from map) not found on disk for word_id {word_id} in XML {xml_file_path.name}.")
                except Exception as e:
                    print(f"Error opening/processing image file {image_full_path} for word_id {word_id}: {e}")
            else:
                pass
    return xml_words_data

# words = []
# words.extend(get_words_from_xml(xml_file) for xml_file in xml_files[:10])

def get_all_words(path: str):
    dataset_path = Path(path)
    
    xml_file_paths_str = sorted([str(p) for p in dataset_path.glob('xml/*.xml')])
    word_image_paths_str = sorted([str(p) for p in dataset_path.glob('words/**/*.png')])

    image_path_map = {Path(p_str).name: Path(p_str) for p_str in word_image_paths_str}
    worker_func = partial(get_words_from_xml_optimized, image_path_map=image_path_map)
        
    num_processes = mp.cpu_count()
    print(num_processes)

    words_list_of_lists = []
    with mp.Pool(processes=num_processes) as pool:
        words_list_of_lists = list(tqdm.tqdm(pool.imap(worker_func, xml_file_paths_str), total=len(xml_file_paths_str), desc="Processing XMLs"))

    words = [word for sublist in words_list_of_lists for word in sublist if sublist]
    return words


class IAMDataset(Dataset):
    def __init__(self, words: list[Word], config: DTrOCRConfig):
        super(IAMDataset, self).__init__()
        self.words = words
        self.processor = DTrOCRProcessor(config, add_eos_token=True, add_bos_token=True)
        
    def __len__(self):
        return len(self.words)
    
    def __getitem__(self, item):
        file_path = self.words[item].file_path
        text = self.words[item].transcription
        
        inputs = self.processor(
            images=Image.open(file_path).convert('RGB'),
            texts=text,
            padding='max_length',
            return_tensors="pt",
            return_labels=True,
        )
        return {
            'pixel_values': inputs.pixel_values[0],
            'input_ids': inputs.input_ids[0],
            'attention_mask': inputs.attention_mask[0],
            'label': text
        }
