# selection/rolling_pivot_selector.py

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from appearance.clip_encoder import CLIPEncoder


def crop_from_bbox(frame: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
    """
    Safely crops a region from a frame using [x1, y1, x2, y2].
    Returns None if the bbox is invalid.
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2]


def bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """
    Returns (cx, cy) of a bbox [x1, y1, x2, y2].
    """
    x1, y1, x2, y2 = bbox
    return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)


def euclidean_distance(p1: Tuple[float, float],
                       p2: Tuple[float, float]) -> float:
    """
    Euclidean distance between two points (x, y).
    """
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two embeddings.
    Returns value in [-1, 1], where 1 means identical direction.
    """
    a = a.reshape(-1)
    b = b.reshape(-1)
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b) / denom)


def track_player_rolling_pivot(
    frames: List[np.ndarray],
    tracks: Dict[str, Any],
    anchor_frame_idx: int,
    anchor_track_id: int,
    encoder: Optional[CLIPEncoder] = None,
    radius_px: int = 150,
    sim_threshold: float = 0.6,
    alpha: float = 0.8,
    max_lost_frames_before_switch: int = 5,
) -> Dict[str, Any]:
    """
    Hybrid tracking of a single player using ByteTrack + rolling-pivot CLIP.

    Now with logic to:
    - Use ByteTrack as the primary source when track_id is available.
    - Use CLIP as fallback when the anchor track_id is temporarily lost.
    - If the anchor id is lost for >= max_lost_frames_before_switch frames,
      and CLIP consistently finds a new candidate, we allow switching the
      anchor_track_id to this new ByteTrack id so that we can trust YOLO again.

    Parameters
    ----------
    frames : list[np.ndarray]
        List of BGR frames from the video.
    tracks : dict
        Output of Tracker.get_object_tracks(frames).
    anchor_frame_idx : int
        Index of the frame where the anchor player is clearly visible.
    anchor_track_id : int
        Initial ByteTrack id for the anchor player.
    encoder : CLIPEncoder, optional
        Shared encoder instance. If None, a new one will be created.
    radius_px : int, default 150
        Spatial radius for CLIP candidate search when track is lost.
    sim_threshold : float, default 0.6
        Minimum cosine similarity to accept CLIP match.
    alpha : float, default 0.8
        Rolling anchor update factor.
    max_lost_frames_before_switch : int, default 5
        Number of consecutive frames without the original anchor_track_id
        before we allow switching the anchor to a new ByteTrack id.

    Returns
    -------
    result : dict with:
        - "bboxes": list[Optional[list]]
        - "track_ids": list[Optional[int]]
        - "used_clip": list[bool]
        - "global_anchor_emb": np.ndarray
        - "params": dict
    """
    if encoder is None:
        encoder = CLIPEncoder()

    num_frames = len(frames)

    # ---- 1) Build global anchor embedding ----
    try:
        anchor_bbox = tracks["players"][anchor_frame_idx][anchor_track_id]["bbox"]
    except KeyError:
        raise ValueError(
            f"Anchor track_id {anchor_track_id} not found in frame {anchor_frame_idx}."
        )

    anchor_frame = frames[anchor_frame_idx]
    anchor_crop = crop_from_bbox(anchor_frame, anchor_bbox)
    if anchor_crop is None:
        raise ValueError(
            "Anchor bbox produced an empty crop; cannot build global anchor embedding."
        )

    global_anchor_emb = encoder.encode(anchor_crop)
    current_anchor_emb = global_anchor_emb.copy()
    current_pos = bbox_center(anchor_bbox)

    # ---- 2) Prepare outputs ----
    hybrid_boxes: List[Optional[List[float]]] = []
    hybrid_track_ids: List[Optional[int]] = []
    used_clip_flags: List[bool] = []

    # Track how many consecutive frames we have lost the current anchor_track_id
    lost_counter = 0

    # ---- 3) Iterate over frames ----
    for i in range(num_frames):
        frame = frames[i]
        player_dict = tracks["players"][i]

        # Case 1: ByteTrack still has the current anchor_track_id
        if anchor_track_id in player_dict:
            bbox = player_dict[anchor_track_id]["bbox"]
            hybrid_boxes.append(bbox)
            hybrid_track_ids.append(anchor_track_id)
            used_clip_flags.append(False)

            current_pos = bbox_center(bbox)
            lost_counter = 0  # reset lost counter because we see the anchor again

            # Optional: update rolling anchor with this frame's appearance
            crop = crop_from_bbox(frame, bbox)
            if crop is not None:
                emb_t = encoder.encode(crop)
                current_anchor_emb = alpha * current_anchor_emb + (1.0 - alpha) * emb_t

        else:
            # Case 2: anchor_track_id is lost -> CLIP-based recovery around last position
            lost_counter += 1

            candidates: List[Tuple[int, List[float]]] = []
            for tid, info in player_dict.items():
                bbox = info["bbox"]
                center = bbox_center(bbox)
                if euclidean_distance(center, current_pos) <= radius_px:
                    candidates.append((tid, bbox))

            best_sim = -1.0
            best_bbox: Optional[List[float]] = None
            best_tid: Optional[int] = None
            best_emb: Optional[np.ndarray] = None

            for tid, bbox in candidates:
                crop = crop_from_bbox(frame, bbox)
                if crop is None:
                    continue
                emb = encoder.encode(crop)
                sim = cosine_sim(current_anchor_emb, emb)
                if sim > best_sim:
                    best_sim = sim
                    best_bbox = bbox
                    best_tid = tid
                    best_emb = emb

            if best_bbox is not None and best_sim >= sim_threshold:
                # Accept CLIP match
                hybrid_boxes.append(best_bbox)
                hybrid_track_ids.append(best_tid)  # this is the actual ByteTrack id for this frame
                used_clip_flags.append(True)

                current_pos = bbox_center(best_bbox)
                # Optionally update rolling anchor even during CLIP fallback
                current_anchor_emb = alpha * current_anchor_emb + (1.0 - alpha) * best_emb

                # 🔁 Key logic: after N lost frames, allow switching anchor_track_id
                if lost_counter >= max_lost_frames_before_switch:
                    anchor_track_id = best_tid
                    lost_counter = 0  # reset, we now trust this as the new anchor id
            else:
                # Player missing in this frame (no good CLIP candidate)
                hybrid_boxes.append(None)
                hybrid_track_ids.append(None)
                used_clip_flags.append(False)
                # lost_counter keeps increasing here

    return {
        "bboxes": hybrid_boxes,
        "track_ids": hybrid_track_ids,
        "used_clip": used_clip_flags,
        "global_anchor_emb": global_anchor_emb,
        "params": {
            "anchor_frame_idx": anchor_frame_idx,
            "anchor_track_id": anchor_track_id,
            "radius_px": radius_px,
            "sim_threshold": sim_threshold,
            "alpha": alpha,
            "max_lost_frames_before_switch": max_lost_frames_before_switch,
        },
    }
