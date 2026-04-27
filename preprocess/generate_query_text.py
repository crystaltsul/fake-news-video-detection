import os
import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
import requests
from dotenv import load_dotenv

# Configuration for datasets
config = [
    'FakeTT',
    'FakeSV',
]
dataset_dir_base = 'data'  # Base directory for datasets

class MyDataset(Dataset):
    """
    Custom Dataset to load video IDs and their corresponding title, transcript, and captions.
    """
    def __init__(self, dataset_dir):
        # vid_file = os.path.join(dataset_dir, 'vids.csv')
        # with open(vid_file, 'r', encoding='utf-8') as f:
        #     self.vids = [line.strip() for line in f if line.strip()]
        self.dataset_dir = dataset_dir
        self.data_df = pd.read_json(os.path.join(dataset_dir, 'data.jsonl'), lines=True, dtype={'vid': str})
        self.caption_df = pd.read_json(os.path.join(dataset_dir, 'retrieve', 'caption.jsonl'), lines=True, dtype={'vid': str})
        self.vids = self.data_df['vid'].tolist()
    def __len__(self):
        return len(self.vids)

    def __getitem__(self, idx):
        vid = self.vids[idx]

        # read data
        try:
            title = self.data_df[self.data_df['vid'] == vid]['title'].iloc[0]
        except Exception as e:
            title = ""
            print(f"Error reading title for vid {vid}: {e}")
        transcript = self.data_df[self.data_df['vid'] == vid]['transcript'].iloc[0]
        
        # read caption
        ocr = self.data_df[self.data_df['vid'] == vid]['ocr'].iloc[0]
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

def generate_integrated_captions():
    """
    Main function to generate integrated captions for each video in the datasets.
    """
    for dataset_name in config:
        print(f"Processing dataset: {dataset_name}")
        
        dataset_dir = os.path.join(dataset_dir_base, dataset_name)
        output_file = os.path.join(dataset_dir, 'retrieve', 'query.jsonl')

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Initialize the dataset and dataloader
        dataset = MyDataset(dataset_dir)
        dataloader = DataLoader(
            dataset,
            batch_size=1,  # Adjust based on your requirements
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0  # Adjust based on your CPU cores
        )
        
        print(f"Starting integrated caption generation for {dataset_name}...")
        
        # Load existing data if the file exists
        if os.path.exists(output_file):
            df_existing = pd.read_json(output_file, lines=True, dtype={'vid': str})
        else:
            df_existing = pd.DataFrame(columns=['vid', 'query', 'intent', 'stance', 'narrative_structure', 'emotional_framing'])
        pbar = tqdm(dataloader, desc=f"For {dataset_name}")
        for batch in pbar:
            vids, titles, transcripts, all_captions = batch
            batch_size = len(vids)
            pbar.set_description(f"For {dataset_name} - Processing {vids[0]}")
            
            for i in range(batch_size):
                vid = vids[i]
                
                # Skip if vid already exists
                if vid in df_existing['vid'].values:
                    continue
                
                title = titles[i]
                transcript = transcripts[i]
                captions = all_captions[i]
                
                # Construct the prompt
                prompt = generate_prompt(title, transcript, captions)
                
                # Call OpenAI API to get integrated caption
                result = call_openai_api(prompt)
                
                # Append new data to the DataFrame
                new_data = pd.DataFrame([{
                    'vid': vid,
                    'query': result.get('summary', ''),
                    'intent': result.get('intent', ''),
                    'stance': result.get('stance', ''),
                    'narrative_structure': result.get('narrative_structure', ''),
                    'emotional_framing': result.get('emotional_framing', '')
                }])
                df_existing = pd.concat([df_existing, new_data], ignore_index=True)
                
                # Save the DataFrame to the file
                df_existing.to_json(output_file, orient='records', lines=True, force_ascii=False)
        
        print(f"Saved integrated captions to {output_file}")
        print(f"Finished processing dataset: {dataset_name}\n")

if __name__ == "__main__":
    load_dotenv()
    generate_integrated_captions()