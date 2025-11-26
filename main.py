from utils.video_utils import read_video, save_video
from utils.map_utils import build_player_heatmap

from tracker import Tracker
from selection.rolling_pivot_selector import track_player_rolling_pivot
from appearance.clip_encoder import CLIPEncoder
import time

from pitch_projection.view_transformer import ViewTransformer
from field.config import CONFIG
from field.draw import draw_pitch, draw_points_on_pitch 

import numpy as np
import pickle
import cv2
import supervision as sv


def main():
    # -----------------------------------------
    # 1. Read video  (limit to 100 frames)
    # -----------------------------------------
    print("[LOG] Reading video...")
    video_frames = read_video("input_videos/test_2.mp4")
    video_frames = video_frames[:350]
    print(f"[LOG] Using {len(video_frames)} frames.")

    # -----------------------------------------
    # 2. Run tracker (YOLO + ByteTrack)
    # -----------------------------------------
    print("[LOG] Initializing tracker...")
    tracker = Tracker('models/tracker/best.pt')

    print("[LOG] Getting tracks...")
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path="stubs/tracks_stub.pkl"
    )

    # -----------------------------------------
    # 3. Produce NORMAL ANNOTATED VIDEO
    # -----------------------------------------
    print("[LOG] Drawing full annotations...")
    annotated_frames = tracker.draw_annotations(video_frames, tracks)

    save_video(annotated_frames, "output_videos/output_annotated.avi")
    print("[LOG] Saved: output_videos/output_annotated.avi")

    # -----------------------------------------
    # 4. Rolling Pivot CLIP: follow only ONE player
    # -----------------------------------------
    print("[LOG] Running Rolling-Pivot CLIP tracking...")

    ANCHOR_FRAME_IDX = 0
    ANCHOR_TRACK_ID = list(tracks["players"][ANCHOR_FRAME_IDX].keys())[2]

    encoder = CLIPEncoder()

    t0 = time.time()
    result = track_player_rolling_pivot(
        video_frames,
        tracks,
        anchor_frame_idx=ANCHOR_FRAME_IDX,
        anchor_track_id=ANCHOR_TRACK_ID,
        encoder=encoder,
        radius_px=150,
        sim_threshold=0.6,
        alpha=0.8,
        max_lost_frames_before_switch=5,
    )
    print(f"[LOG] Rolling Pivot CLIP finished in {time.time() - t0:.2f} seconds.")

    bboxes = result["bboxes"]

    # -----------------------------------------
    # 5. Draw ONLY the Rolling-Pivot player
    # -----------------------------------------
    print("[LOG] Drawing single-player (rolling pivot)...")

    single_player_frames = []
    for frame, bbox in zip(video_frames, bboxes):
        frame_out = frame.copy()

        if bbox is not None:
            frame_out = tracker.draw_ellipse(frame_out, bbox, (0, 255, 0), track_id=777)

        single_player_frames.append(frame_out)

    save_video(single_player_frames, "output_videos/output_single_player_rolling_clip.avi")
    print("[LOG] Saved: output_videos/output_single_player_rolling_clip.avi")

    # -----------------------------------------
    # 6. Build homography once (frame -> pitch)
    # -----------------------------------------
    print("[LOG] Building homography (frame -> pitch)...")

    FIELD_KP_STUB = "stubs/field_keypoints_stub.pkl"
    with open(FIELD_KP_STUB, "rb") as f:
        field_kpts = pickle.load(f)

    kp = field_kpts[ANCHOR_FRAME_IDX]
    if kp is None:
        raise RuntimeError(f"No keypoints for frame {ANCHOR_FRAME_IDX} in field_kpts stub")

    xy = kp["xy"]
    conf = kp["conf"]

    mask = conf > 0.5
    frame_ref = xy[mask]
    pitch_ref = np.array(CONFIG.vertices)[mask]

    transformer = ViewTransformer(
        source=frame_ref.astype(np.float32),
        target=pitch_ref.astype(np.float32)
    )

    # ---------------------------------------------------
    # 7. Generate radar (top-down) frames + trayectoria
    # ---------------------------------------------------
        # ---------------------------------------------------
    # 7. Generate radar (top-down) frames with LIVE HEATMAP
    # ---------------------------------------------------
    print("[LOG] Generating radar (top-down) frames with live heatmap...")

    radar_frames = []
    player_path_pitch = []   # trayectoria acumulada en coords de cancha

    for bbox in bboxes:
        current_pitch_xy = None

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            x = (x1 + x2) / 2
            y = y2
            frame_xy = np.array([[x, y]], dtype=np.float32)

            # proyectamos a la cancha
            pitch_xy = transformer.transform_points(frame_xy)  # (1,2)
            current_pitch_xy = pitch_xy[0]
            player_path_pitch.append(current_pitch_xy)

        # construir heatmap HASTA ESTE FRAME
        if player_path_pitch:
            path_array = np.array(player_path_pitch, dtype=np.float32)
            heat_frame = build_player_heatmap(path_array, CONFIG)
        else:
            heat_frame = draw_pitch(CONFIG)

        # opcional: dibujar el punto actual encima del heatmap
        if current_pitch_xy is not None:
            heat_frame = draw_points_on_pitch(
                config=CONFIG,
                xy=np.array([current_pitch_xy], dtype=np.float32),
                face_color=sv.Color.from_hex("FFFFFF"),  # blanco
                edge_color=sv.Color.BLACK,
                radius=10,
                pitch=heat_frame
            )

        radar_frames.append(heat_frame)


    # ---------------------------------------------------
    # 8. Combine both frames side-by-side into 1 video
    # ---------------------------------------------------
    print("[LOG] Combining frames side-by-side...")

    combined_frames = []

    for left, right in zip(single_player_frames, radar_frames):
        h, w, _ = left.shape
        right_resized = cv2.resize(right, (w, h))
        combined = np.concatenate((left, right_resized), axis=1)
        combined_frames.append(combined)

    save_video(combined_frames, "output_videos/final_single_player_dual_view.avi")
    print("[LOG] Saved: output_videos/final_single_player_dual_view.avi")

    # ---------------------------------------------------
    # 9. Build and save HEATMAP from full trajectory
    # ---------------------------------------------------
    print("[LOG] Building heatmap for player trajectory...")

    if player_path_pitch:
        player_path_pitch_array = np.array(player_path_pitch, dtype=np.float32)
        heatmap_img = build_player_heatmap(player_path_pitch_array, CONFIG)

        cv2.imwrite("output_videos/heatmap_single_player.png", heatmap_img)
        print("[LOG] Saved: output_videos/heatmap_single_player.png")

    print("\n[LOG] All outputs finished!")


if __name__ == "__main__":
    main()
