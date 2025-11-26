# field_keypoints.py

from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import numpy as np


class FieldKeypointDetector:
    def __init__(self, model_path):
        """
        model_path: ruta a tu modelo de keypoints, por ejemplo 'models/field/best.pt'
        """
        self.model = YOLO(model_path)

    def detect_frames(self, frames, batch_size=20, conf=0.3):
        """
        Corre el modelo en batches sobre todos los frames.
        Devuelve la lista de resultados de Ultralytics, uno por frame.
        """
        detections = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            batch_results = self.model.predict(batch, conf=conf, verbose=False)
            detections += batch_results
        return detections

    def get_field_keypoints(self, frames, read_from_stub=False, stub_path=None, conf_min=0.0):
        """
        Devuelve una lista de length = len(frames).
        Cada elemento es:
            - None si no se detectó nada
            - o un dict {"xy": np.ndarray(K,2), "conf": np.ndarray(K,)}
        
        Además guarda/carga un stub con pickle si se indica.
        """
        # 1) Cargar de stub si existe y se pide
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                field_keypoints = pickle.load(f)
            return field_keypoints

        # 2) Si no hay stub, correr el modelo
        results = self.detect_frames(frames)

        field_keypoints = []

        for res in results:
            # Si estás usando YOLO pose para keypoints
            if (not hasattr(res, "keypoints")) or (res.keypoints is None):
                field_keypoints.append(None)
                continue

            # Asumimos solo 1 instancia (la cancha) → index 0
            # Si tu modelo devuelve más de una, luego afinamos la lógica para elegir.
            kpts_xy = res.keypoints.xy[0].cpu().numpy()          # (K, 2)
            kpts_conf = res.keypoints.conf[0].cpu().numpy()      # (K,)

            # Filtramos por confianza mínima si quieres
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

        # 3) Guardar stub si nos dan path
        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, "wb") as f:
                pickle.dump(field_keypoints, f)

        return field_keypoints

    def draw_keypoints(self, frames, field_keypoints, conf_thresh=0.5, radius=6, color=(0, 0, 255)):
        """
        Dibuja los keypoints de la cancha sobre cada frame.
        - frames: lista de frames BGR
        - field_keypoints: lista devuelta por get_field_keypoints
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
