import os
import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import torch
from transformers import CLIPProcessor, CLIPModel

# Configuration for datasets and the CLIP model
config = [
    ['FakeSV', 'openai/clip-vit-large-patch14'],
    ['FakeTT', 'openai/clip-vit-large-patch14'],
]

dataset_dir_base = 'data'  # Base directory for datasets
frames_path = 'frames_32'  # Directory where frames are stored; more frames give better temporal visual coverage for semantic grounding

class MyDataset(Dataset):
    """
    Custom Dataset to load video IDs and their corresponding frames.
    """
    def __init__(self, dataset_dir):
        vid_file = os.path.join(dataset_dir, 'vids.csv')
        with open(vid_file, 'r') as f:
            self.vids = [line.strip() for line in f]
        self.dataset_dir = dataset_dir

    def __len__(self):
        return len(self.vids)

    def __getitem__(self, idx):
        vid = self.vids[idx]
        frames = []
        for i in range(32):  # Assuming 32 frames per video
            frame_path = os.path.join(
                self.dataset_dir,
                frames_path,
                f'{vid}',
                f'frame_{i:03d}.jpg'
            )
            if os.path.exists(frame_path):
                frame = Image.open(frame_path).convert('RGB')
                frames.append(frame)
            else:
                # Substitute missing frames with a black image
                frames.append(Image.new('RGB', (224, 224), color='black'))
        return vid, frames

def collate_fn(batch):
    """
    Custom collate function to prepare batches for the DataLoader.
    """
    vids, all_frames = zip(*batch)  # Unzip the batch
    # Flatten the list of frames
    all_frames = [frame for frames in all_frames for frame in frames]
    return vids, all_frames

def generate_captions():
    """
   Main function to generate CLIP vision-text embeddings and semantic descriptions for each frame.
    """
    for cfg in config:
        dataset_name, model_id = cfg
        print(f"Processing dataset: {dataset_name}")
        
        dataset_dir = os.path.join(dataset_dir_base, dataset_name)
        output_file = os.path.join(dataset_dir, 'retrieve', 'caption.jsonl')

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        print(f"Loading model: {model_id}")
        # Load the CLIP processor and model
        processor = CLIPProcessor.from_pretrained(model_id) 
        model = CLIPModel.from_pretrained(model_id)
        
        # Move model to the appropriate device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Set model to evaluation mode
        model.eval()
        
        # Initialize the dataset and dataloader
        dataset = MyDataset(dataset_dir)
        dataloader = DataLoader(
            dataset,
            batch_size=1,  # Adjust based on your GPU memory
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=8  # Adjust based on your CPU cores
        )
        
        print(f"Starting caption generation for {dataset_name}...")
        
        with torch.no_grad():
            with open(output_file, 'w', encoding='utf-8') as f_out:
                for vids, all_frames in tqdm(dataloader, desc=f"Generating captions for {dataset_name}"):
                    # Move frames to device
                    # CLIP requires both image and text inputs for joint embedding
                    inputs = processor(images=all_frames, text=["a photo of news content"]*len(all_frames), return_tensors="pt", padding=True).to(device)
                    
                    image_features = model.get_image_features(pixel_values=inputs['pixel_values'])
                    
                    # Group captions per video
                    batch_size = len(vids)
                    for i in range(batch_size):
                        vid = vids[i]
                        video_embeddings = image_features[i*32:(i+1)*32].tolist()  # Assuming 32 frames per video
                        json_line = json.dumps({
                            'vid': vid,
                            'clip_embeddings': video_embeddings,
                            'frame_indices': list(range(32))
                        }, ensure_ascii=False)
                        f_out.write(json_line + '\n')
        
        print(f"Saved captions to {output_file}")
        print(f"Finished processing dataset: {dataset_name}\n")

if __name__ == "__main__":
    generate_captions()