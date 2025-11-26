# selection/single_player_selector.py

import numpy as np
from appearance.clip_encoder import CLIPEncoder


def _crop_from_bbox(frame, bbox):
    """
    Recorta el jugador a partir del bbox [x1, y1, x2, y2].
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


def _cosine_distance(a, b):
    """
    Distancia de coseno entre dos embeddings.
    0  → vectores idénticos
    2  → vectores opuestos
    """
    a = a.reshape(-1)
    b = b.reshape(-1)
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    return 1.0 - float(np.dot(a, b) / denom)


def _compute_all_embeddings(video_frames, tracks, encoder):
    """
    Calcula embeddings CLIP para TODOS los jugadores en TODOS los frames.

    Devuelve:
        all_embeddings: lista de dicts por frame:
            all_embeddings[frame_idx][track_id] = embedding (np.array)
    """
    all_embeddings = []

    for frame_idx, frame in enumerate(video_frames):
        players_dict = tracks["players"][frame_idx]
        frame_embeddings = {}

        for track_id, player in players_dict.items():
            bbox = player["bbox"]
            crop = _crop_from_bbox(frame, bbox)
            if crop is None or crop.size == 0:
                continue

            emb = encoder.encode(crop)
            frame_embeddings[track_id] = emb

        all_embeddings.append(frame_embeddings)

    return all_embeddings


def select_single_player(
    video_frames,
    tracks,
    anchor_frame_idx,
    anchor_track_id,
    encoder=None,
    max_distance=None,
):
    """
    Selecciona SIEMPRE al mismo jugador en todos los frames usando CLIP.

    Parámetros
    ----------
    video_frames : list[np.ndarray]
        Lista de frames BGR del video.
    tracks : dict
        Salida de tracker.get_object_tracks(video_frames).
        Se asume el formato:
            tracks["players"][frame_idx][track_id] = {"bbox": [x1, y1, x2, y2], ...}
    anchor_frame_idx : int
        Índice del frame donde eliges al jugador "ancla".
    anchor_track_id : int
        track_id del jugador elegido en ese frame.
    encoder : CLIPEncoder, opcional
        Si no se pasa, se crea uno nuevo.
    max_distance : float, opcional
        Si se especifica, ignora frames donde la mejor coincidencia
        tenga distancia mayor a este valor (devuelve None en ese frame).

    Devuelve
    --------
    single_player_tracks : list[dict | None]
        Lista de longitud = número de frames.
        En cada índice:
            {
              "frame_index": int,
              "track_id": int,
              "bbox": [x1, y1, x2, y2],
              "distance": float
            }
        o None si no se encontró jugador aceptable en ese frame.
    """

    if encoder is None:
        encoder = CLIPEncoder()

    # 1) embeddings de TODOS los jugadores
    all_embeddings = _compute_all_embeddings(video_frames, tracks, encoder)

    # 2) embedding "ancla" del jugador seleccionado
    try:
        anchor_emb = all_embeddings[anchor_frame_idx][anchor_track_id]
    except KeyError:
        raise ValueError(
            f"No se encontró el track_id {anchor_track_id} en el frame {anchor_frame_idx} "
            "para usar como jugador ancla."
        )

    single_player_tracks = []

    # 3) para cada frame, buscamos el jugador más parecido al ancla
    for frame_idx, frame_embeddings in enumerate(all_embeddings):
        if not frame_embeddings:
            single_player_tracks.append(None)
            continue

        best_track = None
        best_dist = float("inf")

        for track_id, emb in frame_embeddings.items():
            dist = _cosine_distance(anchor_emb, emb)
            if dist < best_dist:
                best_dist = dist
                best_track = track_id

        if best_track is None:
            single_player_tracks.append(None)
            continue

        # si hay umbral, filtramos
        if (max_distance is not None) and (best_dist > max_distance):
            single_player_tracks.append(None)
            continue

        bbox = tracks["players"][frame_idx][best_track]["bbox"]

        single_player_tracks.append(
            {
                "frame_index": frame_idx,
                "track_id": int(best_track),
                "bbox": bbox,
                "distance": float(best_dist),
            }
        )

    return single_player_tracks
