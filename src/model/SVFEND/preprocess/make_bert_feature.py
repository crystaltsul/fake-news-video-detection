from transformers import CLIPModel, CLIPTokenizer, ChineseCLIPModel, BertTokenizer
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from tqdm import tqdm
import os
import numpy as np
import torch
import torch.nn as nn

config = [
    ['FakeSV', 'OFA-Sys/chinese-clip-vit-large-patch14'],
    ['FakeTT', 'openai/clip-vit-large-patch14']
]

dataset_dir_base = 'data/'

class MyTextDataset(Dataset):
    def __init__(self, dataset_dir):
        self.data_df = pd.read_json(os.path.join(dataset_dir, 'data.jsonl'), lines=True, dtype={'vid': 'str'})

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        vid = self.data_df.loc[index, 'vid']
        
        title = self.data_df.loc[index, 'title']
        ocr = self.data_df.loc[index, 'ocr']
        trans = self.data_df.loc[index, 'transcript']
        
        text = title + ' ' + ocr + ' ' + trans

        return vid, text

def collate_fn(batch):
    vids, texts = zip(*batch)
    return vids, texts

for cfg in config:
    dataset_name, model_id = cfg
    print(f"Processing dataset: {dataset_name}")
    

    dataset_dir = os.path.join(dataset_dir_base, dataset_name)
    output_file = os.path.join(dataset_dir, 'fea', 'SVFEND/fea_clip_text.pt')

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Loading model: {model_id}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if 'chinese' in model_id.lower():
        tokenizer = BertTokenizer.from_pretrained(model_id)
        model = ChineseCLIPModel.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    else:
        tokenizer = CLIPTokenizer.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id, device_map='auto')

    model.eval()

    dataset = MyTextDataset(dataset_dir)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=2)
    
    features = {}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Encoding texts for {dataset_name}"):
            vids, texts = batch

            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=77).to(device)

            cls_text = model.get_text_features(**inputs)
            cls_text = cls_text.float().cpu()
            
            for i, vid in enumerate(vids):
                features[vid] = cls_text[i]

    print(f"Saving features to {output_file}")
    torch.save(features, output_file)
    print(f"Finished processing dataset: {dataset_name}\n")