📌 README.md — Single-Player Heatmap Generator from Broadcast Football Video
🟧 Single-Player Heatmap Generator
Broadcast Football Video → Player Tracking → Homography → Heatmap

This project implements a full single-camera football analysis system that extracts the trajectory of one selected player from broadcast footage and generates a top-down heatmap of their movement throughout the clip.

It combines:

YOLOv8 (player detection)

ByteTrack (multi-object temporal tracking)

CLIP Vision Transformer (Rolling-Pivot identity stabilization)

Image preprocessing (CLAHE, Canny, Hough)

YOLOv8 field-keypoint model

Homography estimation

Top-down pitch projection & heatmap accumulation

📁 Project Structure
.
├── input_videos/
│   └── test_2.mp4
├── models/
│   ├── tracker/best.pt               # YOLOv8 player-tracking model  (NOT INCLUDED)
│   ├── field_keypoints/best.pt       # YOLOv8 keypoint model         (NOT INCLUDED)
│   └── clip/                         # CLIP ViT downloaded automatically
├── tracker/
│   ├── tracker.py
│   └── __init__.py
├── selection/
│   ├── rolling_pivot_selector.py
│   ├── single_player_selector.py
│   └── __init__.py
├── appearance/
│   ├── clip_encoder.py
│   └── __init__.py
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
│   └── field_keypoints_stub.pkl
├── output_videos/
│   ├── output_annotated.avi
│   ├── output_single_player_rolling_clip.avi
│   └── final_single_player_dual_view.avi
├── tests/
│   └── VitTests.ipynb
├── main.py
└── README.md

⚠️ Missing Files (IMPORTANT)

This repository does not include trained YOLO models due to size restrictions.

You must manually download and place:

Model	Expected Path	Purpose
best.pt	models/tracker/best.pt	YOLOv8 model to detect players, goalkeepers, referees and the ball
best.pt	models/field_keypoints/best.pt	YOLOv8 model to detect field line intersections for homography

If these files are missing, the pipeline will not run.

🔧 Installation
1. Create environment
conda create -n heatmap python=3.10
conda activate heatmap

2. Install dependencies
pip install ultralytics supervision opencv-python numpy matplotlib tqdm transformers pillow scipy


CLIP ViT will download automatically on first run.

▶️ Running the system

Simply run:

python main.py


The following outputs will be generated inside /output_videos/:

Output file	Description
output_annotated.avi	YOLOv8 detections + ByteTrack ellipses
output_single_player_rolling_clip.avi	Only the selected player (Rolling-Pivot CLIP)
final_single_player_dual_view.avi	Broadcast + top-down radar side-by-side
heatmap_single_player.png	Final heatmap image of the player's movement
🧠 Pipeline Overview
1. Player Detection & Tracking

YOLOv8 detects players, referees, goalkeepers and ball.

ByteTrack assigns track IDs per frame.

2. Rolling-Pivot CLIP ViT Identity Stabilization

A ViT-B/32 encoder generates embeddings for each player crop.

This solves:

ID switches

Occlusions

Fast motion

Merges & splits

Mechanism:

Select initial player (anchor frame & track ID).

Extract global anchor embedding with CLIP.

Each new valid match updates a rolling anchor.

If YOLO/ByteTrack loses the player →
use CLIP to select the most similar nearby detection.

After N missing frames → ID can smoothly switch.

This provides a stable logical identity even when tracking fails.

🎯 Field Keypoint Detection & Homography

The pitch is detected using:

CLAHE (contrast enhancement)

Canny edges

Hough lines

YOLOv8 keypoint model (field intersections)

Homography is computed using RANSAC:

(image_x, image_y) → (pitch_x, pitch_y)


This enables:

Metric trajectories

Heatmap spatial accuracy

True pitch-coordinate visualization

🔥 Heatmap Generation

Convert bbox → player foot position

Project point to pitch

Accumulate over time

Draw pitch + gaussian density map

Overlay current position (dual-view mode)

The user gets a clean 2D density heatmap of player movement.

📈 Ablation Study (Provided in Poster)

The project includes experiments comparing:

Preprocessing methods

Raw

CLAHE + Sobel

CLAHE + Canny + Hough

Tracking stability

Baseline ByteTrack only

Rolling-Pivot CLIP ViT

Metrics evaluated:

Valid keypoints

Homography accuracy

Missing frames

Center jump

ID switches

Recovered occlusions

💡 Notes & Limitations

This system tracks only one player at a time.

Ball detection is not fully robust (small object limitation).

Homography depends heavily on field-line visibility.

Broadcast zooming can introduce metric distortions.

📌 Future Work

Multi-player & team-level heatmaps

Automatic player-of-interest selection

Better ball detector

Optical-flow smoothing of ground-truth trajectory

Real-time version (GPU-only pipeline)

🤝 Contributing

Pull requests are welcome.
For major changes, please open an issue first to discuss your proposal.

📄 License

MIT License.
