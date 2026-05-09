import torch
import pandas as pd
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

DATASETS = ['FakeSV', 'FakeTT']
INTENT_WEIGHT = 0.6      # weight for intent similarity
CONTENT_WEIGHT = 0.4     # weight for content similarity
INTRA_DOMAIN_BOOST = 0.05  # small boost for same-dataset references
TOPK = 100

def load_features(datasets):
    """Load intent and content features for all datasets into unified dicts."""
    intent_fea, content_fea, labels_dict, domain_dict = {}, {}, {}, {}
    for ds in datasets:
        path = Path('data') / ds
        intent = torch.load(path / 'retrieve' / 'fea_retrieval_intent.pt', weights_only=True)
        content = torch.load(path / 'retrieve' / 'fea_retrieval_content.pt', weights_only=True)
        labels = pd.read_json(path / 'label.jsonl', lines=True, dtype={'vid': str})
        ldict = labels.set_index('vid')['label'].to_dict()
        for vid in intent:
            intent_fea[vid] = intent[vid]
            content_fea[vid] = content[vid]
            domain_dict[vid] = ds
        labels_dict.update(ldict)
    return intent_fea, content_fea, labels_dict, domain_dict

def compute_dual_similarities(query_ids, base_ids, intent_fea, content_fea,
                               labels_dict, domain_dict, query_domain,
                               topk=TOPK):
    """
    For each query video, compute weighted intent+content similarity against
    all base videos, apply intra-domain boost, then return top-K real and fake.
    """
    q_intent = np.stack([intent_fea[v].numpy() for v in query_ids])
    q_content = np.stack([content_fea[v].numpy() for v in query_ids])
    b_intent = np.stack([intent_fea[v].numpy() for v in base_ids])
    b_content = np.stack([content_fea[v].numpy() for v in base_ids])

    # Normalise for cosine similarity
    def norm(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    q_intent, b_intent = norm(q_intent), norm(b_intent)
    q_content, b_content = norm(q_content), norm(b_content)

    intent_sim = q_intent @ b_intent.T    # (num_query, num_base)
    content_sim = q_content @ b_content.T
    combined = INTENT_WEIGHT * intent_sim + CONTENT_WEIGHT * content_sim

    # Apply intra-domain boost
    for j, base_vid in enumerate(base_ids):
        if domain_dict.get(base_vid) == query_domain:
            combined[:, j] += INTRA_DOMAIN_BOOST

    base_id_to_idx = {vid: idx for idx, vid in enumerate(base_ids)}

    results = []
    for i, qvid in enumerate(tqdm(query_ids, desc=f'Retrieving {query_domain}')):
        sim = combined[i].copy()
        # Mask self
        if qvid in base_id_to_idx:
            sim[base_id_to_idx[qvid]] = -np.inf

        results_0, results_1 = [], []
        for idx in np.argsort(sim)[::-1]:
            bvid = base_ids[idx]
            label = labels_dict.get(bvid, -1)
            if label == 0 and len(results_0) < topk:
                results_0.append({'vid': bvid, 'similarity': float(sim[idx])})
            elif label == 1 and len(results_1) < topk:
                results_1.append({'vid': bvid, 'similarity': float(sim[idx])})
            if len(results_0) >= topk and len(results_1) >= topk:
                break

        results.append({
            'vid': qvid,
            'similarities': [
                {'vid': [r['vid'] for r in results_0], 'sim': [r['similarity'] for r in results_0]},
                {'vid': [r['vid'] for r in results_1], 'sim': [r['similarity'] for r in results_1]},
            ]
        })
    return results

# Load unified cross-domain pool
intent_fea, content_fea, labels_dict, domain_dict = load_features(DATASETS)

for dataset in DATASETS:
    path = Path('data') / dataset
    train_vids = pd.read_csv(path / 'vids' / 'vid_time3_train.txt', header=None, dtype=str)[0].tolist()
    valid_vids = pd.read_csv(path / 'vids' / 'vid_time3_valid.txt', header=None, dtype=str)[0].tolist()
    test_vids  = pd.read_csv(path / 'vids' / 'vid_time3_test.txt',  header=None, dtype=str)[0].tolist()

    all_vids = [v for v in set(train_vids + valid_vids + test_vids) if v in labels_dict]

    # Base pool: train+valid from BOTH datasets (cross-domain)
    other_dataset = [d for d in DATASETS if d != dataset][0]
    other_path = Path('data') / other_dataset
    other_train = pd.read_csv(other_path / 'vids' / 'vid_time3_train.txt', header=None, dtype=str)[0].tolist()
    other_valid = pd.read_csv(other_path / 'vids' / 'vid_time3_valid.txt', header=None, dtype=str)[0].tolist()
    other_base = [v for v in set(other_train + other_valid) if v in labels_dict]

    intra_base = [v for v in set(train_vids + valid_vids) if v in labels_dict]
    base_pool = list(set(intra_base + other_base))  # unified cross-domain pool

    results = compute_dual_similarities(
        all_vids, base_pool, intent_fea, content_fea,
        labels_dict, domain_dict, query_domain=dataset
    )

    output_path = path / 'retrieve' / 'sim.jsonl'
    with open(output_path, 'w') as f:
        for item in results:
            f.write(json.dumps(item) + '\n')
    print(f'Saved {dataset} → {output_path}')