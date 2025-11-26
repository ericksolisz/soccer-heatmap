# Single-Player Heatmap Generator  
### Broadcast Football Video → Player Tracking → Homography → Heatmap

This project implements a complete single-camera computer vision pipeline that extracts a single player's trajectory from a broadcast football match and generates a top-down movement heatmap.

It uses a combination of:

- YOLOv8 for player detection  
- ByteTrack for multi-object tracking  
- A Rolling-Pivot CLIP ViT module to stabilize player identity  
- CLAHE + Canny + Hough for field-line enhancement  
- A YOLOv8 keypoint detector for field intersections  
- Homography estimation to map frame coordinates to pitch coordinates  
- A top-down pitch renderer and density-based heatmap generator  

---

## Project Structure

```
.
├── input_videos/
│   └── test_2.mp4
├── models/
│   ├── tracker/best.pt               # YOLOv8 player-tracker (NOT INCLUDED)
│   ├── field_keypoints/best.pt       # YOLOv8 field keypoint model (NOT INCLUDED)
│   └── clip/                         # CLIP ViT downloads automatically
├── tracker/
│   └── tracker.py
├── selection/
│   ├── rolling_pivot_selector.py
│   ├── single_player_selector.py
│   └── __init__.py
├── appearance/
│   └── clip_encoder.py
├── utils/
│   ├── video_utils.py
│   ├── map_utils.py
│   ├── bbox_utils.py
│   └── __init__.py
├── pitch_projection/
│   ├── view_transformer.py
│   └── ...
├── field/
│   ├── config.py
│   ├── draw.py
│   └── ...
├── stubs/
│   ├── tracks_stub.pkl
│   ├── field_keypoints_stub.pkl
├── output_videos/
│   ├── output_annotated.avi
│   ├── output_single_player_rolling_clip.avi
│   └── final_single_player_dual_view.avi
├── tests/
│   └── VitTests.ipynb
├── main.py
└── README.md
```

---

## Important: Missing Models

The YOLO models **are not included** in this repository for size reasons.

You must provide these files:

| Required Model | Expected Path | Description |
|----------------|---------------|-------------|
| `best.pt` | `models/tracker/best.pt` | YOLOv8 model trained to detect players/referees/goalkeepers/ball |
| `best.pt` | `models/field_keypoints/best.pt` | YOLOv8 model trained to detect field intersections for homography |

Without these files the pipeline will not run.

---

## Installation

### Create environment

```bash
conda create -n heatmap python=3.10
conda activate heatmap
```

### Install dependencies

```bash
pip install ultralytics supervision opencv-python numpy matplotlib tqdm transformers pillow scipy
```

CLIP ViT will be downloaded automatically on first run.

---

## Running the System

Run:

```bash
python main.py
```

The following outputs will appear in `output_videos/`:

| File | Description |
|------|-------------|
| `output_annotated.avi` | Broadcast video with YOLOv8 detections + ByteTrack |
| `output_single_player_rolling_clip.avi` | Only the selected player using ViT stabilization |
| `final_single_player_dual_view.avi` | Side-by-side broadcast + top-down radar view |
| `heatmap_single_player.png` | Final spatial density heatmap |

---

## Pipeline Overview

### 1. Player Detection & Tracking
- YOLOv8 detects players, referees, goalkeepers and the ball  
- ByteTrack assigns temporal IDs  

### 2. Rolling-Pivot CLIP ViT Identity Stabilization
Solves:
- ID switches  
- Occlusions  
- Player overlap  
- Fast movement  

Mechanism:
1. Extract embedding of the selected player (global anchor).  
2. Maintain a rolling anchor updated each frame.  
3. If YOLO/ByteTrack loses the player, search nearby candidates.  
4. Select highest cosine similarity (CLIP ViT-B/32).  
5. After N missing frames, allow smooth identity switch.  

### 3. Field Keypoint Detection & Homography
- CLAHE → Canny → Hough for line enhancement  
- YOLOv8 keypoint model predicts pitch intersections  
- RANSAC computes homography (frame → pitch coordinates)  

### 4. Heatmap Generation
1. Convert bbox → player foot coordinate  
2. Project to pitch  
3. Accumulate trajectory  
4. Render Gaussian density over a pitch template  

Final output: **a clean 2D heatmap of player movement**  

---

## Ablation Study (Summary)

### Field-line preprocessing

| Method | Valid Keypoints | Homography Accuracy |
|--------|-----------------|---------------------|
| Raw Frame | 68% | 88% |
| CLAHE + Sobel | 82% | 93% |
| CLAHE + Canny + Hough | **91%** | **97%** |

### Tracking Stability

| Metric | Baseline | Rolling-Pivot ViT |
|--------|----------|-------------------|
| Missing frames | 34 | **8** |
| Avg center jump | 29.1 px | **12.4 px** |
| ID switches | 5 | **1** |
| Recovered occlusions | 0% | **92%** |

---

## Limitations

- Ball detection is still challenging at long distances  
- Homography depends on field-line visibility  
- Tracks only a single player per sequence  
- Broadcast zooming may distort pitch geometry  

---

## Future Work

- Multi-player & team-level heatmaps  
- Better ball detection  
- Real-time version (GPU-only)  
- Optical-flow smoothing  
- Automatic player-of-interest selection  

---

## License

MIT License.

