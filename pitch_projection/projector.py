# pitch_projection/projector.py

import os
import pickle
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np

from .view_transformer import ViewTransformer 


@dataclass
class SimplePitchConfig:
    """
    Cancha 2D simple en pixeles (top-down).
    width_px  = largo horizontal de la cancha en la vista top-down
    height_px = alto vertical
    vertices  = lista de puntos de referencia en ese sistema de coordenadas.
                Aquí ponemos simplemente las 4 esquinas.
    """
    width_px: int = 1050   # por ejemplo 10px por metro si piensas en 105m
    height_px: int = 680   # 10px por metro si piensas en 68m

    def __post_init__(self):
        # orden: esquina superior izquierda, superior derecha, inferior derecha, inferior izquierda
        self.vertices = np.array([
            [0, 0],
            [self.width_px, 0],
            [self.width_px, self.height_px],
            [0, self.height_px]
        ], dtype=np.float32)


class PitchProjector:
    def __init__(self, config: SimplePitchConfig, transformer: ViewTransformer):
        self.config = config
        self.transformer = transformer

    # ---------- FACTORY: construir desde stub de keypoints ----------
    @classmethod
    def from_field_keypoints_stub(
        cls,
        field_keypoints_stub_path: str,
        config: SimplePitchConfig,
        calib_frame_idx: int = 0,
        corner_indices: List[int] = None,
        conf_thresh: float = 0.5
    ) -> "PitchProjector":
        """
        Crea un PitchProjector cargando los keypoints de cancha de un stub
        y calculando la homografía con ViewTransformer.

        field_keypoints_stub_path:
            stub que generaste en generate_field_keypoints_video.py
        calib_frame_idx:
            frame que usas para calibrar (toma donde se vea bien la cancha)
        corner_indices:
            índices de los keypoints que corresponden a las 4 esquinas
            en el orden: [top_left, top_right, bottom_right, bottom_left]
        """
        if not os.path.exists(field_keypoints_stub_path):
            raise FileNotFoundError(f"No se encontró stub de keypoints: {field_keypoints_stub_path}")

        with open(field_keypoints_stub_path, "rb") as f:
            field_keypoints = pickle.load(f)

        kp = field_keypoints[calib_frame_idx]
        if kp is None:
            raise RuntimeError(f"No hay keypoints en calib_frame_idx={calib_frame_idx}")

        xy_all = kp["xy"]   # (K, 2)
        conf_all = kp["conf"]

        if corner_indices is None:
            # ⚠️ IMPORTANTE:
            # Aquí asumimos que tus 4 primeras keypoints son las esquinas.
            # AJUSTA ESTO según cómo esté entrenado tu modelo.
            corner_indices = [0, 1, 2, 3]

        xy_img = []
        xy_pitch = []

        for i_corner, idx in enumerate(corner_indices):
            if idx >= xy_all.shape[0]:
                raise RuntimeError(
                    f"corner index {idx} fuera de rango para el número de keypoints ({xy_all.shape[0]})"
                )
            if conf_all[idx] < conf_thresh:
                print(f"[WARN] keypoint {idx} (corner {i_corner}) tiene conf baja ({conf_all[idx]:.2f})")

            xy_img.append(xy_all[idx])

        xy_img = np.array(xy_img, dtype=np.float32)   # (4,2)
        # target: las 4 esquinas de nuestra imagen de cancha
        xy_pitch = config.vertices[:4].astype(np.float32)  # (4,2)

        transformer = ViewTransformer(source=xy_img, target=xy_pitch)
        return cls(config=config, transformer=transformer)

    # ---------- PROYECCIÓN DE PUNTOS ----------
    def project_points_frame_to_pitch(self, points_xy: np.ndarray) -> np.ndarray:
        """
        points_xy: array (N, 2) en coordenadas de imagen (pixeles).
        Devuelve (N, 2) en coordenadas de cancha (pixeles del mapa top-down).
        """
        if points_xy.size == 0:
            return points_xy
        return self.transformer.transform_points(points_xy.astype(np.float32))

    def project_players_from_tracks(
        self,
        tracks: Dict[str, Any],
        frame_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Usa tu estructura de tracks:
          tracks["players"][frame_idx][track_id]["bbox"] = [x1, y1, x2, y2]

        Devuelve:
          - player_ids: np.ndarray (N,) con los track_ids
          - pitch_xy: np.ndarray (N,2) en coordenadas top-down
        """
        players_dict = tracks["players"][frame_idx]

        if len(players_dict) == 0:
            return np.array([]), np.zeros((0, 2), dtype=np.float32)

        img_points = []
        ids = []

        for track_id, data in players_dict.items():
            bbox = data["bbox"]  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox

            x_center = (x1 + x2) / 2.0
            y_bottom = y2

            img_points.append([x_center, y_bottom])
            ids.append(track_id)

        img_points = np.array(img_points, dtype=np.float32)
        pitch_xy = self.project_points_frame_to_pitch(img_points)
        ids = np.array(ids)

        return ids, pitch_xy

    # ---------- DIBUJAR MAPA TOP-DOWN ----------
    def draw_pitch_base(self) -> np.ndarray:
        """
        Crea una imagen verde con líneas blancas de la cancha.
        """
        w, h = self.config.width_px, self.config.height_px
        pitch = np.zeros((h, w, 3), dtype=np.uint8)

        # fondo verde
        pitch[:, :] = (40, 120, 40)  # BGR

        # borde blanco
        cv2.rectangle(
            pitch,
            (0, 0),
            (w - 1, h - 1),
            (255, 255, 255),
            thickness=4
        )

        # línea de medio campo
        cv2.line(
            pitch,
            (w // 2, 0),
            (w // 2, h - 1),
            (255, 255, 255),
            thickness=2
        )

        # un circulito central para que se vea bonito
        cv2.circle(
            pitch,
            (w // 2, h // 2),
            60,
            (255, 255, 255),
            thickness=2
        )

        return pitch

    def draw_players_on_pitch(
        self,
        pitch_img: np.ndarray,
        pitch_xy: np.ndarray,
        color=(0, 0, 255),
        radius: int = 10
    ) -> np.ndarray:
        """
        Dibuja players como puntitos en el mapa top-down.
        pitch_xy: (N, 2) en coords de cancha (pixeles).
        """
        out = pitch_img.copy()

        for (x, y) in pitch_xy:
            cv2.circle(
                out,
                (int(x), int(y)),
                radius,
                color,
                thickness=-1
            )

        return out
