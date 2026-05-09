import os
import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
import requests
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration for datasets
config = [
    'FakeTT',
    'FakeSV',
]
MAX_WORKERS = 10
dataset_dir_base = 'data'  # Base directory for datasets

class MyDataset(Dataset):
    """
    Custom Dataset to load video IDs and their corresponding title, transcript, and captions.
    """
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        data_df = pd.read_json(os.path.join(dataset_dir, 'data.jsonl'), lines=True, dtype={'vid': str})
        # Pre-index by vid for O(1) lookup instead of O(n) filter per item
        self.vid_to_row = data_df.set_index('vid').to_dict('index')
        self.vids = data_df['vid'].tolist()

    def __len__(self):
        return len(self.vids)

    def __getitem__(self, idx):
        vid = self.vids[idx]
        row = self.vid_to_row[vid]
        title = row.get('title', '')
        transcript = row.get('transcript', '')
        ocr = row.get('ocr', '')
        captions = ocr if ocr else "[No visual text detected]"
        return vid, title, transcript, captions

def collate_fn(batch):
    """
    Custom collate function to prepare batches for the DataLoader.
    """
    vids, titles, transcripts, all_captions = zip(*batch)
    return vids, titles, transcripts, all_captions

def generate_prompt(title, transcript, captions):
    """
    Constructs a prompt for Llama-3 to extract deep semantic dimensions
    from multimodal news video content: intent, stance, narrative structure,
    and emotional framing — rather than surface-level content description.
    """
    prompt = f"""You are an expert in multimodal misinformation analysis.

You are given the following content from a news video:

Video Title: {title}
Audio Transcript: {transcript}
Visual Scene Descriptions: {captions}

Analyse this video across the following four dimensions:

1. INTENT: What is this content trying to make the viewer believe or feel? Is it attempting to inform, persuade, deceive, or manipulate?
2. STANCE: How is the subject matter framed? Is the framing neutral, alarmist, one-sided, or misleading? Does the visual content support or contradict the textual claims?
3. NARRATIVE STRUCTURE: How is the story being constructed? Does it use selective emphasis, omission of context, false causality, or recontextualisation of real footage?
4. EMOTIONAL FRAMING: What emotional response is being targeted (e.g. fear, outrage, urgency, sympathy)? How do the visuals and language work together to amplify this?

Respond in the following JSON format:
{{
  "intent": "...",
  "stance": "...",
  "narrative_structure": "...",
  "emotional_framing": "...",
  "summary": "A single sentence beginning with 'The video attempts to' that captures the overall manipulative or informational strategy."
}}"""
    return prompt

def call_openai_api(prompt, max_retries=5, backoff_factor=2):
    """
    Calls the OpenAI API with the given prompt and handles retries using the requests library.
    """
    url = os.getenv('OPENAI_API_URL')
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json"
    }
    data = {
        "model": os.getenv('OPENAI_MODEL'),
        "messages": [
            {"role": "system", "content": """You are an expert in multimodal misinformation analysis. \
You respond only in valid JSON with no additional text, preamble, or markdown formatting. \
Your response must always contain exactly these keys: intent, stance, narrative_structure, emotional_framing, summary. \
Each value must be a single concise string of 1-3 sentences."""},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 800,
        "temperature": 0.2
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            raw = response.json()['choices'][0]['message']['content'].strip()
            parsed = json.loads(raw)
            required_keys = {"intent", "stance", "narrative_structure", "emotional_framing", "summary"}
            if not required_keys.issubset(parsed.keys()):
                missing = required_keys - parsed.keys()
                print(f"Warning: Missing keys in response: {missing}. Filling with empty strings.")
                for key in missing:
                    parsed[key] = ""
            return parsed
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                wait_time = backoff_factor ** attempt
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"HTTP error: {e}. Retrying...")
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}. Retrying...")
        except json.JSONDecodeError:
            # Try stripping markdown code fences before giving up
            try:
                clean = raw.replace('```json', '').replace('```', '').strip()
                parsed = json.loads(clean)
                return parsed
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON response. Retrying...")
        except Exception as e:
            print(f"Unexpected error: {e}. Skipping this item.")
            return {"intent": "", "stance": "", "narrative_structure": "", "emotional_framing": "", "summary": ""}
    return {"intent": "", "stance": "", "narrative_structure": "", "emotional_framing": "", "summary": ""}

def generate_content_prompt(title, transcript, captions):
    """Generates a content-description query for topical similarity retrieval."""
    return f"""You are a multimodal news video analyst.

Given the following video content:
Title: {title}
Audio Transcript: {transcript}
Visual Scene Descriptions: {captions}

Write a single concise sentence (max 30 words) describing what this video is about — 
the topic, event, people, and location depicted. Focus on factual content only.

Respond with only the sentence, no preamble."""

def call_openai_api_simple(prompt, max_retries=5, backoff_factor=2):
    url = os.getenv('OPENAI_API_URL')
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json"
    }
    data = {
        "model": os.getenv('OPENAI_MODEL'),
        "messages": [
            {"role": "system", "content": "You are a factual news video analyst. Respond with a single concise sentence only, no preamble."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 100,
        "temperature": 0.1
    }
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            time.sleep(backoff_factor ** attempt)
    return ""

def process_single_video(vid, title, transcript, captions):
    """Calls both APIs concurrently for a single video."""
    with ThreadPoolExecutor(max_workers=2) as executor:
        intent_future = executor.submit(
            call_openai_api, generate_prompt(title, transcript, captions)
        )
        content_future = executor.submit(
            call_openai_api_simple, generate_content_prompt(title, transcript, captions)
        )
        intent_result = intent_future.result()
        content_result = content_future.result()
    return intent_result, content_result

def process_video_batch(pending_items):
    def process_one(item):
        vid, title, transcript, captions = item
        intent_result, content_result = process_single_video(vid, title, transcript, captions)
        return {
            'vid': vid,
            'query': intent_result.get('summary', ''),
            'intent': intent_result.get('intent', ''),
            'stance': intent_result.get('stance', ''),
            'narrative_structure': intent_result.get('narrative_structure', ''),
            'emotional_framing': intent_result.get('emotional_framing', ''),
            'content_query': content_result,
        }
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_one, item): item for item in pending_items}
        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing videos'):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error processing video: {e}")
    return results

def generate_integrated_captions():
    for dataset_name in config:
        print(f"Processing dataset: {dataset_name}")

        dataset_dir = os.path.join(dataset_dir_base, dataset_name)
        output_file = os.path.join(dataset_dir, 'retrieve', 'query.jsonl')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        dataset = MyDataset(dataset_dir)

        if os.path.exists(output_file):
            df_existing = pd.read_json(output_file, lines=True, dtype={'vid': str})
        else:
            df_existing = pd.DataFrame(columns=['vid', 'query', 'intent', 'stance',
                                                 'narrative_structure', 'emotional_framing'])

        existing_vids = set(df_existing['vid'].values)
        has_content = set()
        if 'content_query' in df_existing.columns:
            has_content = set(
                df_existing[
                    df_existing['content_query'].notna() &
                    (df_existing['content_query'] != '')
                ]['vid'].values
            )

        # Collect videos needing full processing vs content_query only
        pending_full = []
        pending_content_only = []

        for idx in range(len(dataset)):
            vid, title, transcript, captions = dataset[idx]
            if vid in has_content:
                continue
            elif vid in existing_vids:
                pending_content_only.append((vid, title, transcript, captions))
            else:
                pending_full.append((vid, title, transcript, captions))

        print(f"  {len(pending_full)} videos need full processing")
        print(f"  {len(pending_content_only)} videos need content_query only")

        # Process full videos in parallel
        if pending_full:
            results = process_video_batch(pending_full)
            new_df = pd.DataFrame(results)
            df_existing = pd.concat([df_existing, new_df], ignore_index=True)
            df_existing.to_json(output_file, orient='records', lines=True, force_ascii=False)
            print(f"  Saved {len(results)} new videos")

        # Process content-only videos in parallel
        if pending_content_only:
            def get_content_only(item):
                vid, title, transcript, captions = item
                content = call_openai_api_simple(
                    generate_content_prompt(title, transcript, captions)
                )
                return vid, content

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(get_content_only, item): item
                           for item in pending_content_only}
                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc='Content queries'):
                    vid, content = future.result()
                    df_existing.loc[df_existing['vid'] == vid, 'content_query'] = content

            df_existing.to_json(output_file, orient='records', lines=True, force_ascii=False)
            print(f"  Updated content_query for {len(pending_content_only)} videos")

        print(f"Finished processing dataset: {dataset_name}\n")

# def generate_integrated_captions():
#     """
#     Main function to generate integrated captions for each video in the datasets.
#     """
#     for dataset_name in config:
#         print(f"Processing dataset: {dataset_name}")
        
#         dataset_dir = os.path.join(dataset_dir_base, dataset_name)
#         output_file = os.path.join(dataset_dir, 'retrieve', 'query.jsonl')

#         os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
#         # Initialize the dataset and dataloader
#         dataset = MyDataset(dataset_dir)
#         dataloader = DataLoader(
#             dataset,
#             batch_size=1,  # Adjust based on your requirements
#             shuffle=False,
#             collate_fn=collate_fn,
#             num_workers=0  # Adjust based on your CPU cores
#         )
        
#         print(f"Starting integrated caption generation for {dataset_name}...")
        
#         # Load existing data if the file exists
#         if os.path.exists(output_file):
#             df_existing = pd.read_json(output_file, lines=True, dtype={'vid': str})
#         else:
#             df_existing = pd.DataFrame(columns=['vid', 'query', 'intent', 'stance', 'narrative_structure', 'emotional_framing'])
#         pbar = tqdm(dataloader, desc=f"For {dataset_name}")
#         for batch in pbar:
#             vids, titles, transcripts, all_captions = batch
#             batch_size = len(vids)
#             pbar.set_description(f"For {dataset_name} - Processing {vids[0]}")
            
#             for i in range(batch_size):
#                 vid = vids[i]
#                 title = titles[i]
#                 transcript = transcripts[i]
#                 captions = all_captions[i]
                
#                 # Skip if vid already exists
#                 if vid in df_existing['vid'].values:
#                     existing_row = df_existing[df_existing['vid'] == vid]
#                     if 'content_query' in existing_row.columns and pd.notna(existing_row['content_query'].iloc[0]) and existing_row['content_query'].iloc[0] != '':
#                         continue
#                     # Otherwise fall through to generate content_query only
#                     intent_result = {
#                         'summary': existing_row['query'].iloc[0],
#                         'intent': existing_row['intent'].iloc[0],
#                         'stance': existing_row['stance'].iloc[0],
#                         'narrative_structure': existing_row['narrative_structure'].iloc[0],
#                         'emotional_framing': existing_row['emotional_framing'].iloc[0],
#                     }
#                     content_result = call_openai_api_simple(generate_content_prompt(title, transcript, captions))
#                     df_existing.loc[df_existing['vid'] == vid, 'content_query'] = content_result
#                     df_existing.to_json(output_file, orient='records', lines=True, force_ascii=False)
#                     continue
                
#                 # Generate BOTH queries
#                 intent_result = call_openai_api(generate_prompt(title, transcript, captions))
#                 # content_result = call_openai_api(generate_content_prompt(title, transcript, captions))
#                 content_result = call_openai_api_simple(generate_content_prompt(title, transcript, captions))
                
#                 new_data = pd.DataFrame([{
#                     'vid': vid,
#                     'query': intent_result.get('summary', ''),
#                     'intent': intent_result.get('intent', ''),
#                     'stance': intent_result.get('stance', ''),
#                     'narrative_structure': intent_result.get('narrative_structure', ''),
#                     'emotional_framing': intent_result.get('emotional_framing', ''),
#                     'content_query': content_result,
#                 }])
#                 df_existing = pd.concat([df_existing, new_data], ignore_index=True)
                
#                 # Save the DataFrame to the file
#                 df_existing.to_json(output_file, orient='records', lines=True, force_ascii=False)
        
#         print(f"Saved integrated captions to {output_file}")
#         print(f"Finished processing dataset: {dataset_name}\n")

if __name__ == "__main__":
    load_dotenv()
    generate_integrated_captions()