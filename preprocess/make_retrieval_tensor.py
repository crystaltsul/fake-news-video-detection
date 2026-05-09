from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from tqdm import tqdm
import os
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

# REPLACE entire config and dataset class:

config = [
    'FakeSV',
    'FakeTT',
]
MODEL_NAME = 'intfloat/multilingual-e5-large'  # single encoder for both datasets

def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

class MyTextDataset(Dataset):
    def __init__(self, dataset_dir):
        self.data_df = pd.read_json(
            os.path.join(dataset_dir, 'retrieve', 'query.jsonl'),
            lines=True, dtype={'vid': 'str'}
        )

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        row = self.data_df.iloc[index]
        vid = row['vid']
        # Intent query — manipulation strategy description
        intent_text = (
            f"Intent: {row['intent']} "
            f"Stance: {row['stance']} "
            f"Narrative: {row['narrative_structure']} "
            f"Emotional framing: {row['emotional_framing']}"
        )
        # Content query — topical description
        content_text = row.get('content_query', row.get('query', ''))
        return vid, intent_text, content_text

class CollateClass:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        vids, intent_texts, content_texts = zip(*batch)
        intent_inputs = self.tokenizer(
            list(intent_texts), padding=True, truncation=True,
            max_length=512, return_tensors='pt'
        )
        content_inputs = self.tokenizer(
            list(content_texts), padding=True, truncation=True,
            max_length=512, return_tensors='pt'
        )
        return vids, intent_inputs, content_inputs

# Main loop — single encoder for all datasets
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

for dataset_name in config:
    dataset_dir = f'data/{dataset_name}'
    intent_output = os.path.join(dataset_dir, 'retrieve', 'fea_retrieval_intent.pt')
    content_output = os.path.join(dataset_dir, 'retrieve', 'fea_retrieval_content.pt')
    os.makedirs(os.path.dirname(intent_output), exist_ok=True)

    dataset = MyTextDataset(dataset_dir)
    collate_fn = CollateClass(tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False,
                            collate_fn=collate_fn, num_workers=4)

    intent_features = {}
    content_features = {}

    with torch.no_grad():
        for vids, intent_inputs, content_inputs in tqdm(dataloader, desc=dataset_name):
            intent_inputs = intent_inputs.to(device)
            content_inputs = content_inputs.to(device)

            intent_out = model(**intent_inputs)
            content_out = model(**content_inputs)

            intent_emb = average_pool(intent_out.last_hidden_state, intent_inputs['attention_mask']).cpu()
            content_emb = average_pool(content_out.last_hidden_state, content_inputs['attention_mask']).cpu()

            for i, vid in enumerate(vids):
                intent_features[vid] = intent_emb[i]
                content_features[vid] = content_emb[i]

    torch.save(intent_features, intent_output)
    torch.save(content_features, content_output)
    print(f"Saved {dataset_name}: {len(intent_features)} intent + content vectors")


# config = [
#     ('FakeSV', 'thenlper/gte-large-zh'),
#     ('FakeTT', 'thenlper/gte-large'),
# ]

# def average_pool(last_hidden_states: Tensor,
#                  attention_mask: Tensor) -> Tensor:
#     last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
#     return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


# class MyTextDataset(Dataset):
#     def __init__(self, dataset_dir):
#         # self.data_df = pd.read_json(os.path.join(dataset_dir, 'lm_ocr.jsonl'), lines=True, dtype={'vid': 'str'})
#         self.data_df = pd.read_json(os.path.join(dataset_dir, 'retrieve', 'query.jsonl'), lines=True, dtype={'vid': 'str'})

#     def __len__(self):
#         return len(self.data_df)

#     def __getitem__(self, index):
#         vid = self.data_df.loc[index, 'vid']
#         row = self.data_df.iloc[index]
        
#         text = (
#             f"Intent: {row['intent']} "
#             f"Stance: {row['stance']} "
#             f"Narrative: {row['narrative_structure']} "
#             f"Emotional framing: {row['emotional_framing']}"
#         )
#         return vid, text

# class CollateClass:
#     def __init__(self, tokenizer):
#         self.tokenizer = tokenizer
        
#     def __call__(self, batch):
#         vids, texts = zip(*batch)
#         inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
#         return vids, inputs

# for dataset_name, model_name in config:
#     dataset_dir = f'data/{dataset_name}'
#     output_file = os.path.join(dataset_dir, 'retrieve', 'fea_retrieval_semantic.pt')

#     os.makedirs(os.path.dirname(output_file), exist_ok=True)

#     print(f"Loading model: {model_name}")
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModel.from_pretrained(model_name, device_map='cuda' if torch.cuda.is_available() else 'cpu')

#     dataset = MyTextDataset(dataset_dir)
#     collate_fn = CollateClass(tokenizer)
#     dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=8)

#     features = {}

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#     model.eval()

#     with torch.no_grad():
#         for batch in tqdm(dataloader, desc=f"Encoding texts for {dataset_name}"):
#             vids, inputs = batch
            
#             inputs = inputs.to(device)
#             outputs = model(**inputs)
            
#             # Get embeddings from the last hidden state
#             # embeddings = outputs.last_hidden_state[:, 0, :]  # Take [CLS] token embedding
#             embeddings = average_pool(outputs.last_hidden_state, inputs['attention_mask'])
#             embeddings = embeddings.cpu()

#             for i, vid in enumerate(vids):
#                 features[vid] = embeddings[i]
#     # print length of features
#     print(f"Length of features: {len(features)}")
#     print(f"Saving features to {output_file}")
#     torch.save(features, output_file)
#     print(f"Finished processing dataset: {dataset_name}\n")