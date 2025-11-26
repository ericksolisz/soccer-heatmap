import numpy as np
import cv2
from field.draw import draw_pitch

def build_player_heatmap(pitch_points, config, padding=50, scale=0.1):
    """
    pitch_points: array (N, 2) en coords de cancha (cm, como CONFIG.vertices)
    Devuelve una imagen BGR con el heatmap sobre la cancha.
    """
    # 1) Base de cancha
    pitch_img = draw_pitch(config, padding=padding, scale=scale)
    h, w, _ = pitch_img.shape

    # 2) Mapa de calor crudo
    heat = np.zeros((h, w), dtype=np.float32)

    for (x, y) in pitch_points:
        xi = int(x * scale) + padding
        yi = int(y * scale) + padding

        if 0 <= xi < w and 0 <= yi < h:
            heat[yi, xi] += 1.0

    # 3) Difuminar para que no se vea solo puntos
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=15, sigmaY=15)

    if heat.max() > 0:
        heat = heat / heat.max()  # normalizar 0..1

    # 4) Construir mapa amarillo → rojo (BGR)
    heat_bgr = np.zeros_like(pitch_img, dtype=np.uint8)

    t = heat  # (h, w)

    # De amarillo (0,255,255) a rojo (0,0,255)
    heat_bgr[..., 0] = 0                                  # B
    heat_bgr[..., 1] = (255 * (1.0 - t)).astype(np.uint8) # G
    heat_bgr[..., 2] = 255                                # R

    # 5) En zonas donde no hay nada, que no pinte
    mask = heat > 1e-2
    overlay = pitch_img.copy()
    alpha = 0.6

    overlay[mask] = cv2.addWeighted(
        pitch_img[mask], 1 - alpha,
        heat_bgr[mask], alpha,
        0
    )

    return overlay
