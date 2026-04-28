"""
make_event_features.py  —  Stage 2: Dynamic Temporal Pseudo-Pair Generation

Input : data/{dataset}/fea/SVFEND/clip_visual_features.pt
            dict { vid -> Tensor(32, 768) }   (produced by make_vgg19_feature.py)

Output: data/{dataset}/fea/SVFEND/event_features.pt
            dict { vid -> Tensor(NUM_EVENTS, 768) }

Each video's 32 CLIP frame embeddings are segmented into temporal narrative
events via cosine-distance shot-boundary detection. Frames within each
detected shot are mean-pooled into a single event vector. The resulting
fixed-length sequence of event vectors (padded / truncated to NUM_EVENTS)
is what the model attends over — preserving narrative order rather than
treating the video as a bag of frames.

Why this matters:
    CRAVE and REAL both collapse retrieved videos to single vectors before
    prototype construction. By keeping the event dimension intact we allow
    the TemporalProtoGenerator (in SVFEND_model.py) to build prototypes
    that are sensitive to *where* in the narrative a manipulation occurs,
    enabling temporal-aware contrastive learning.
"""

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

config = [
    'FakeSV',
    'FakeTT',
]

dataset_dir_base = 'data'
NUM_FRAMES = 32         # frames per video in clip_visual_features.pt
NUM_EVENTS = 8          # fixed output length (shots padded / truncated to this)
SHOT_THRESHOLD = 0.25   # cosine-distance threshold for declaring a shot boundary
                        # lower  → more sensitive, more shots detected
                        # higher → fewer, coarser segments
                        # 0.25 is a reasonable default for CLIP embeddings;
                        # tune on a small held-out set if needed


# ---------------------------------------------------------------------------
# Shot boundary detection
# ---------------------------------------------------------------------------

def detect_shot_boundaries(frame_embeddings: torch.Tensor,
                           threshold: float = SHOT_THRESHOLD) -> list[int]:
    """
    Detect shot boundaries in a sequence of normalised frame embeddings.

    Args:
        frame_embeddings: Tensor of shape (T, D), one embedding per frame.
        threshold:        Cosine-distance threshold. Consecutive frames whose
                          cosine distance exceeds this are considered a new shot.

    Returns:
        List of boundary indices (inclusive start of each new shot), always
        starting with 0.  E.g. [0, 8, 19] means shots span frames 0–7,
        8–18, and 19–31.
    """
    T = frame_embeddings.shape[0]
    if T <= 1:
        return [0]

    normed = F.normalize(frame_embeddings.float(), dim=-1)  # (T, D)

    # Cosine similarity between consecutive frames  →  cosine distance
    cos_sim = (normed[:-1] * normed[1:]).sum(dim=-1)        # (T-1,)
    cos_dist = 1.0 - cos_sim                                # (T-1,)

    boundaries = [0]
    for i, dist in enumerate(cos_dist):
        if dist.item() > threshold:
            boundaries.append(i + 1)   # frame i+1 starts a new shot

    return boundaries


# ---------------------------------------------------------------------------
# Event pooling
# ---------------------------------------------------------------------------

def pool_shots_to_events(frame_embeddings: torch.Tensor,
                         boundaries: list[int],
                         num_events: int = NUM_EVENTS) -> torch.Tensor:
    """
    Mean-pool frames within each shot to produce one event vector per shot,
    then pad or truncate the shot sequence to exactly `num_events` vectors.

    Padding strategy: replicate the last event vector (avoids introducing
    zero-padding noise into the attention mechanism).

    Args:
        frame_embeddings: Tensor (T, D).
        boundaries:       Shot boundary indices from detect_shot_boundaries.
        num_events:       Desired fixed output length.

    Returns:
        Tensor of shape (num_events, D).
    """
    T, D = frame_embeddings.shape
    shot_vectors = []

    for idx, start in enumerate(boundaries):
        end = boundaries[idx + 1] if idx + 1 < len(boundaries) else T
        shot_frames = frame_embeddings[start:end]           # (shot_len, D)
        shot_vectors.append(shot_frames.mean(dim=0))        # (D,)

    event_tensor = torch.stack(shot_vectors, dim=0)         # (num_shots, D)
    num_shots = event_tensor.shape[0]

    if num_shots >= num_events:
        # Too many shots — keep the first (num_events - 1) and the last one
        # to preserve both the opening and closing narrative context.
        indices = list(range(num_events - 1)) + [num_shots - 1]
        event_tensor = event_tensor[indices]                # (num_events, D)

    else:
        # Too few shots — pad by replicating the final event vector
        pad_len = num_events - num_shots
        last = event_tensor[-1].unsqueeze(0).expand(pad_len, -1)
        event_tensor = torch.cat([event_tensor, last], dim=0)  # (num_events, D)

    assert event_tensor.shape == (num_events, D), \
        f"Unexpected shape after pooling: {event_tensor.shape}"

    return event_tensor


# ---------------------------------------------------------------------------
# Per-video processing
# ---------------------------------------------------------------------------

def compute_event_features(frame_features: dict[str, torch.Tensor],
                            num_events: int = NUM_EVENTS,
                            threshold: float = SHOT_THRESHOLD
                            ) -> dict[str, torch.Tensor]:
    """
    Process every video in `frame_features` and return a new dict mapping
    vid -> Tensor(num_events, D).

    Args:
        frame_features: dict loaded from clip_visual_features.pt,
                        { vid: Tensor(32, 768) }.
        num_events:     Fixed output event count.
        threshold:      Shot boundary detection threshold.

    Returns:
        dict { vid: Tensor(num_events, 768) }
    """
    event_features = {}

    for vid, fea in tqdm(frame_features.items(), desc='Segmenting videos'):
        # fea: (32, 768)  on CPU (weights_only load)
        boundaries = detect_shot_boundaries(fea, threshold=threshold)
        event_vec  = pool_shots_to_events(fea, boundaries, num_events=num_events)
        event_features[vid] = event_vec     # (num_events, 768)

    return event_features


# ---------------------------------------------------------------------------
# Diagnostics helper (optional but useful during development)
# ---------------------------------------------------------------------------

def print_segmentation_stats(frame_features: dict[str, torch.Tensor],
                              threshold: float = SHOT_THRESHOLD) -> None:
    """
    Print a brief summary of detected shot counts across the dataset.
    Helps calibrate SHOT_THRESHOLD before committing to a full run.
    """
    shot_counts = []
    for fea in frame_features.values():
        boundaries = detect_shot_boundaries(fea, threshold=threshold)
        shot_counts.append(len(boundaries))

    shot_counts_t = torch.tensor(shot_counts, dtype=torch.float)
    print(f"  Threshold = {threshold}")
    print(f"  Videos    = {len(shot_counts)}")
    print(f"  Shots per video — min: {shot_counts_t.min().item():.0f}  "
          f"mean: {shot_counts_t.mean().item():.1f}  "
          f"max: {shot_counts_t.max().item():.0f}")
    under = (shot_counts_t < 2).sum().item()
    over  = (shot_counts_t > NUM_EVENTS).sum().item()
    print(f"  Videos with <2 shots (no boundary detected): {under}")
    print(f"  Videos with >{NUM_EVENTS} shots (will be truncated): {over}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    for dataset_name in config:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*60}")

        fea_dir   = os.path.join(dataset_dir_base, dataset_name, 'fea', 'SVFEND')
        input_file  = os.path.join(fea_dir, 'clip_visual_features.pt')
        output_file = os.path.join(fea_dir, 'event_features.pt')

        # Skip if already computed
        if os.path.exists(output_file):
            print(f"  event_features.pt already exists — skipping.")
            print(f"  Delete {output_file} to recompute.")
            continue

        if not os.path.exists(input_file):
            print(f"  ERROR: {input_file} not found.")
            print(f"  Run make_vgg19_feature.py first.")
            continue

        print(f"  Loading clip_visual_features.pt ...")
        frame_features = torch.load(input_file, weights_only=True)
        print(f"  Loaded {len(frame_features)} videos,  "
              f"frame tensor shape: {next(iter(frame_features.values())).shape}")

        # Print calibration stats before running
        print(f"\n  Shot segmentation diagnostics (threshold={SHOT_THRESHOLD}):")
        print_segmentation_stats(frame_features, threshold=SHOT_THRESHOLD)

        print(f"\n  Computing event features  "
              f"(NUM_EVENTS={NUM_EVENTS}, threshold={SHOT_THRESHOLD}) ...")
        event_features = compute_event_features(
            frame_features,
            num_events=NUM_EVENTS,
            threshold=SHOT_THRESHOLD,
        )

        # Sanity check output shape
        sample_vid = next(iter(event_features))
        sample_shape = event_features[sample_vid].shape
        assert sample_shape == (NUM_EVENTS, 768), \
            f"Expected ({NUM_EVENTS}, 768), got {sample_shape}"

        os.makedirs(fea_dir, exist_ok=True)
        torch.save(event_features, output_file)
        print(f"  Saved {len(event_features)} videos → {output_file}")
        print(f"  Output tensor shape per video: {sample_shape}")

    print("\nDone.")


if __name__ == '__main__':
    main()
