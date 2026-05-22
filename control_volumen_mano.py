"""
=============================================================
  CONTROL DE VOLUMEN CON GESTOS DE MANO
  Compatible con mediapipe 0.10.30+  |  macOS / Windows / Linux
  Uso: python3 control_volumen_mano.py
=============================================================
  - Mostrá la mano frente a la cámara
  - Juntá pulgar e índice  → volumen BAJO
  - Separá pulgar e índice → volumen ALTO
  - Presioná 'q' para salir
=============================================================
"""

import cv2
import numpy as np
import math
import subprocess
import sys
import platform
import urllib.request
import os
import threading

# ─── Descargar modelo si no existe ───────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("[...] Descargando modelo (~8 MB), espera un momento...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[OK] Modelo descargado.")
    except Exception as e:
        print(f"[ERROR] No se pudo descargar el modelo: {e}")
        print(f"  Descargalo manualmente desde:\n  {MODEL_URL}")
        print(f"  y ponelo junto al script como '{MODEL_PATH}'")
        sys.exit(1)
else:
    print("[OK] Modelo listo.")

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

# ─── Control de volumen (en hilo separado para no bloquear el video) ──────────
OS = platform.system()
_vol_lock   = threading.Lock()
_vol_target = 50
_vol_actual = -1   # fuerza el primer set

def _volume_worker():
    """Hilo que aplica el volumen solo cuando cambia, sin bloquear el loop."""
    global _vol_actual
    while True:
        with _vol_lock:
            target = _vol_target
        if target != _vol_actual:
            _apply_volume(target)
            _vol_actual = target
        import time; time.sleep(0.05)   # revisa 20 veces por segundo

def _apply_volume(pct: int):
    pct = max(0, min(100, int(pct)))
    if OS == "Darwin":
        subprocess.run(["osascript", "-e", f"set volume output volume {pct}"],
                       capture_output=True)
    elif OS == "Windows":
        try:
            vol_db = _VOL_MIN + (_VOL_MAX - _VOL_MIN) * (pct / 100.0)
            _vol_ctrl.SetMasterVolumeLevel(vol_db, None)
        except Exception:
            pass
    elif OS == "Linux":
        subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{pct}%"],
                       capture_output=True)

def set_volume(pct: int):
    """Pide cambiar el volumen (no-bloqueante)."""
    global _vol_target
    with _vol_lock:
        _vol_target = max(0, min(100, int(pct)))

if OS == "Darwin":
    print("[OK] macOS: osascript listo.")
elif OS == "Windows":
    try:
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        _devices   = AudioUtilities.GetSpeakers()
        _interface = _devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        _vol_ctrl  = cast(_interface, POINTER(IAudioEndpointVolume))
        _VOL_MIN, _VOL_MAX = _vol_ctrl.GetVolumeRange()[:2]
        print("[OK] Windows: pycaw listo.")
    except ImportError:
        print("[!] Instalá: pip install pycaw comtypes"); sys.exit(1)
elif OS == "Linux":
    print("[OK] Linux: pactl listo.")
else:
    print(f"[!] OS no soportado: {OS}"); sys.exit(1)

# Arrancar hilo de volumen (daemon = se cierra solo al salir)
_t = threading.Thread(target=_volume_worker, daemon=True)
_t.start()

# ─── Resultado compartido (callback asíncrono de mediapipe) ───────────────────
latest_landmarks = []

def on_result(result, output_image, timestamp_ms):
    global latest_landmarks
    latest_landmarks = result.hand_landmarks[0] if result.hand_landmarks else []

# ─── HandLandmarker ───────────────────────────────────────────────────────────
options = HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.5,
    result_callback=on_result,
)
detector = HandLandmarker.create_from_options(options)

# ─── Parámetros ───────────────────────────────────────────────────────────────
DIST_MIN    = 20
DIST_MAX    = 220
SMOOTHING   = 0.2
FONT        = cv2.FONT_HERSHEY_SIMPLEX
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# ─── Cámara ───────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] No se pudo abrir la cámara."); sys.exit(1)

# Resolución moderada para mejor FPS
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
cap.set(cv2.CAP_PROP_FPS, 60)

print("\n[LISTO] Mostrá tu mano frente a la cámara. Presioná 'q' para salir.\n")

vol_suavizado  = 50.0
volumen_actual = 50
timestamp      = 0

# FPS counter
import time
fps_t  = time.time()
fps_c  = 0
fps    = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] No se pudo leer el frame."); break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    # Enviar al detector (escala reducida para más velocidad)
    small  = cv2.resize(frame, (480, 270))
    rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp += 1
    detector.detect_async(mp_img, timestamp)

    # Dibujar landmarks (coordenadas escaladas al frame original)
    if latest_landmarks:
        lm = latest_landmarks
        x_thumb = int(lm[4].x * w);  y_thumb = int(lm[4].y * h)
        x_index = int(lm[8].x * w);  y_index = int(lm[8].y * h)

        for a, b in CONNECTIONS:
            cv2.line(frame,
                     (int(lm[a].x*w), int(lm[a].y*h)),
                     (int(lm[b].x*w), int(lm[b].y*h)),
                     (180, 180, 180), 1)
        for i in range(21):
            cv2.circle(frame, (int(lm[i].x*w), int(lm[i].y*h)),
                       4, (100, 200, 255), cv2.FILLED)

        dist          = math.hypot(x_index - x_thumb, y_index - y_thumb)
        vol_raw       = np.interp(dist, [DIST_MIN, DIST_MAX], [0, 100])
        vol_suavizado = vol_suavizado * (1-SMOOTHING) + vol_raw * SMOOTHING
        volumen_actual = int(vol_suavizado)
        set_volume(volumen_actual)

        cx = (x_thumb + x_index) // 2
        cy = (y_thumb + y_index) // 2
        cv2.circle(frame, (x_thumb, y_thumb), 14, (255, 0, 255), cv2.FILLED)
        cv2.circle(frame, (x_index, y_index), 14, (255, 0, 255), cv2.FILLED)
        cv2.line(frame, (x_thumb, y_thumb), (x_index, y_index), (0, 255, 0), 3)
        cv2.circle(frame, (cx, cy), 9, (0, 255, 0), cv2.FILLED)

    bx1, bx2, by_top, by_bot = 40, 75, 130, 380
    bfill = int(np.interp(volumen_actual, [0, 100], [by_bot, by_top]))
    cv2.rectangle(frame, (bx1, by_top), (bx2, by_bot), (60, 60, 60), cv2.FILLED)
    cv2.rectangle(frame, (bx1, bfill),  (bx2, by_bot), (0, 215, 255), cv2.FILLED)
    cv2.rectangle(frame, (bx1, by_top), (bx2, by_bot), (160, 160, 160), 2)
    cv2.putText(frame, "VOL", (bx1-2, by_top-12), FONT, 0.55, (255,255,255), 2)
    cv2.putText(frame, f"{volumen_actual}%", (bx1-2, by_bot+28), FONT, 0.65, (0,215,255), 2)

    fps_c += 1
    if time.time() - fps_t >= 1.0:
        fps   = fps_c
        fps_c = 0
        fps_t = time.time()


    cv2.imshow("Control de Volumen con Mano", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()
print("\n[FIN] Cerrando correctamente.")