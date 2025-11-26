
from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import numpy as np
from .preprocessing import preprocess_frames


class FieldKeypointDetector:
    def __init__(self, model_path):
        """
        model_path: path to your keypoint model, e.g. 'models/field_keypoints/best.pt'
        """
        self.model = YOLO(model_path)

    def detect_frames(self, frames, batch_size=20, conf=0.3):
        """
        Run the YOLOv8 pose model on all frames in batches.
        Returns a list of Ultralytics results, one per frame.
        """
        detections = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            batch_results = self.model.predict(batch, conf=conf, verbose=False)
            detections += batch_results
        return detections

    def get_field_keypoints(
        self,
        frames,
        read_from_stub=False,
        stub_path=None,
        conf_min=0.0,
        apply_preprocessing=True,
    ):
        """
        Returns a list with length = len(frames).
        Each element is:
            - None if nothing was detected
            - or a dict {"xy": np.ndarray(K,2), "conf": np.ndarray(K,)}

        Optionally:
        - Load/save a stub (pickle) with the keypoints.
        - Apply preprocessing (CLAHE + Canny + Hough) before detection.
        """

        # 1) Load from stub if requested and available
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                field_keypoints = pickle.load(f)
            return field_keypoints

        # 2) Preprocess frames if requested
        if apply_preprocessing:
            frames_for_model = preprocess_frames(frames)
        else:
            frames_for_model = frames

        # 3) Run model on (possibly preprocessed) frames
        results = self.detect_frames(frames_for_model)

        field_keypoints = []

        for res in results:
            # If you're using YOLO pose for keypoints
            if (not hasattr(res, "keypoints")) or (res.keypoints is None):
                field_keypoints.append(None)
                continue

            # Assume only 1 instance (the pitch) → index 0
            kpts_xy = res.keypoints.xy[0].cpu().numpy()       # (K, 2)
            kpts_conf = res.keypoints.conf[0].cpu().numpy()   # (K,)

            # Confidence filtering (optional)
            if conf_min > 0.0:
                mask = kpts_conf >= conf_min
                kpts_xy = kpts_xy[mask]
                kpts_conf = kpts_conf[mask]

            if kpts_xy.shape[0] == 0:
                field_keypoints.append(None)
            else:
                field_keypoints.append({
                    "xy": kpts_xy,
                    "conf": kpts_conf
                })

        # 4) Save stub if path provided
        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, "wb") as f:
                pickle.dump(field_keypoints, f)

        return field_keypoints

    def draw_keypoints(self, frames, field_keypoints, conf_thresh=0.5, radius=6, color=(0, 0, 255)):
        """
        Draw the field keypoints on each frame.
        - frames: list of BGR frames (original ones)
        - field_keypoints: list returned by get_field_keypoints
        """
        output_frames = []

        for frame, kp in zip(frames, field_keypoints):
            frame_out = frame.copy()

            if kp is not None:
                xy = kp["xy"]
                conf = kp["conf"]

                for (x, y), c in zip(xy, conf):
                    if c < conf_thresh:
                        continue
                    cv2.circle(
                        frame_out,
                        (int(x), int(y)),
                        radius,
                        color,
                        thickness=-1
                    )

            output_frames.append(frame_out)

        return output_frames
