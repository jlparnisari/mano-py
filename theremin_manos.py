"""
=============================================================
  🎵 THEREMIN CON DOS MANOS
  Uso: python3 theremin_manos.py

  MANO DERECHA   → altura de la mano = TONO
  MANO IZQUIERDA → distancia pulgar-índice = VOLUMEN
  Presioná 'q' para salir
=============================================================
"""

import cv2
import numpy as np
import math
import sys
import os
import urllib.request
import threading
import time
import subprocess
import wave
import tempfile

SAMPLE_RATE = 22050
CHUNK_SECS  = 0.12   # chunks largos = sin gaps

class ThereminSynth:
    def __init__(self):
        self._freq  = 440.0
        self._vol   = 0.0
        self._tf    = 440.0
        self._tv    = 0.0
        self._phase = 0.0
        self._lock  = threading.Lock()
        self._running = True
        # Dos hilos alternados para audio sin cortes
        self._procs = [None, None]
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def set(self, freq, vol):
        with self._lock:
            self._tf = max(60.0,  min(2000.0, float(freq)))
            self._tv = max(0.0,   min(1.0,    float(vol)))

    def _make_wav(self, freq, vol):
        n     = int(SAMPLE_RATE * CHUNK_SECS)
        omega = 2 * math.pi * freq / SAMPLE_RATE
        t     = np.arange(n, dtype=np.float64)
        w     = np.sin(self._phase + omega * t)
        w    += 0.4  * np.sin(self._phase*2 + omega*2*t)
        w    += 0.15 * np.sin(self._phase*3 + omega*3*t)
        w    /= 1.55
        self._phase = (self._phase + omega * n) % (2 * math.pi)
        s = (w * vol * 28000).astype(np.int16)
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        with wave.open(tmp.name, 'wb') as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
            wf.writeframes(s.tobytes())
        return tmp.name

    def _loop(self):
        files = [None, None]
        slot  = 0
        while self._running:
            with self._lock:
                tf = self._tf; tv = self._tv

            self._freq = self._freq * 0.7 + tf * 0.3
            self._vol  = self._vol  * 0.7 + tv * 0.3

            if self._vol < 0.02:
                # Silencio — matar procesos activos
                for p in self._procs:
                    if p and p.poll() is None:
                        p.terminate()
                self._procs = [None, None]
                for f in files:
                    if f:
                        try: os.unlink(f)
                        except: pass
                files = [None, None]
                time.sleep(0.05)
                continue

            # Generar siguiente chunk
            path = self._make_wav(self._freq, self._vol)

            # Esperar a que el slot anterior termine
            prev = self._procs[slot]
            if prev and prev.poll() is None:
                prev.wait()

            # Limpiar archivo anterior de este slot
            if files[slot]:
                try: os.unlink(files[slot])
                except: pass

            # Lanzar nuevo afplay
            proc = subprocess.Popen(
                ['afplay', path],
                stderr=subprocess.DEVNULL
            )
            self._procs[slot] = proc
            files[slot]       = path
            slot = 1 - slot   # alternar 0 y 1

    def stop(self):
        self._running = False
        for p in self._procs:
            if p:
                try: p.terminate()
                except: pass
        subprocess.run(['pkill', '-f', 'afplay'], capture_output=True)

synth = ThereminSynth()
time.sleep(0.2)

# ─── Modelo mediapipe ─────────────────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

if not os.path.exists(MODEL_PATH):
    print("[...] Descargando modelo (~8 MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[OK] Modelo descargado.")
    except Exception as e:
        print(f"[ERROR] {e}"); sys.exit(1)
else:
    print("[OK] Modelo listo.")

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

hands_data = {}
data_lock  = threading.Lock()

def on_result(result, output_image, timestamp_ms):
    with data_lock:
        hands_data.clear()
        if result.hand_landmarks:
            for i, lm in enumerate(result.hand_landmarks):
                try:
                    label = result.handedness[i][0].display_name
                    hands_data[label] = lm
                except: pass

options = HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.5,
    result_callback=on_result,
)
detector = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] No se pudo abrir la cámara."); sys.exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
cap.set(cv2.CAP_PROP_FPS, 30)

print("\n[LISTO] Mostrá tus dos manos.")
print("  MANO DERECHA   → altura = TONO  (arriba=agudo, abajo=grave)")
print("  MANO IZQUIERDA → separá pulgar e índice = VOLUMEN")
print("  Presioná 'q' para salir.\n")

FONT = cv2.FONT_HERSHEY_SIMPLEX
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

def draw_hand(frame, lm, w, h, cj, cb):
    for a, b in CONNECTIONS:
        cv2.line(frame,
                 (int(lm[a].x*w), int(lm[a].y*h)),
                 (int(lm[b].x*w), int(lm[b].y*h)), cb, 2)
    for i in range(21):
        cv2.circle(frame, (int(lm[i].x*w), int(lm[i].y*h)), 6, cj, cv2.FILLED)

timestamp   = 0
freq_smooth = 440.0
vol_smooth  = 0.0
DIST_MIN    = 20
DIST_MAX    = 200
NOTE_NAMES  = ["Do","Do#","Re","Re#","Mi","Fa","Fa#","Sol","Sol#","La","La#","Si"]
WAVE_LEN    = 240
wave_hist   = np.zeros(WAVE_LEN, dtype=np.float32)
t_wave      = 0.0
fps_t = time.time(); fps_c = 0; fps = 0

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    small  = cv2.resize(frame, (480, 270))
    rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp += 1
    detector.detect_async(mp_img, timestamp)

    with data_lock:
        right_hand = hands_data.get("Left")   # flip → mano derecha usuario
        left_hand  = hands_data.get("Right")  # flip → mano izquierda usuario

    freq_target = freq_smooth
    vol_target  = 0.0  # si no hay mano izquierda, silencio

    # MANO DERECHA → TONO por altura
    if right_hand:
        lm = right_hand
        tips_y = np.mean([lm[4].y, lm[8].y, lm[12].y, lm[16].y, lm[20].y])
        freq_target = float(np.interp(tips_y, [0.05, 0.92], [1800.0, 60.0]))
        draw_hand(frame, lm, w, h, cj=(0,220,255), cb=(0,130,210))
        cx = int(lm[9].x * w); cy = int(lm[9].y * h)
        cv2.putText(frame, f"{int(freq_smooth)} Hz", (cx-45, cy-25),
                    FONT, 0.7, (0,220,255), 2)

    # MANO IZQUIERDA → VOLUMEN por distancia pulgar-índice
    if left_hand:
        lm = left_hand
        x_thumb = int(lm[4].x * w); y_thumb = int(lm[4].y * h)
        x_index = int(lm[8].x * w); y_index = int(lm[8].y * h)
        dist       = math.hypot(x_index - x_thumb, y_index - y_thumb)
        vol_target = float(np.interp(dist, [DIST_MIN, DIST_MAX], [0.0, 1.0]))
        draw_hand(frame, lm, w, h, cj=(100,255,120), cb=(50,190,70))
        cx_m = (x_thumb + x_index) // 2
        cy_m = (y_thumb + y_index) // 2
        cv2.circle(frame, (x_thumb, y_thumb), 13, (255,80,255), cv2.FILLED)
        cv2.circle(frame, (x_index, y_index), 13, (255,80,255), cv2.FILLED)
        cv2.line(frame, (x_thumb, y_thumb), (x_index, y_index), (255,80,255), 3)
        cv2.putText(frame, f"Vol {int(vol_smooth*100)}%", (cx_m-40, cy_m-20),
                    FONT, 0.7, (100,255,120), 2)

    freq_smooth = freq_smooth * 0.8 + freq_target * 0.2
    vol_smooth  = vol_smooth  * 0.8 + vol_target  * 0.2
    synth.set(freq_smooth, vol_smooth)

    # Onda visual
    t_wave += 1.0 / 30.0
    sample  = math.sin(2 * math.pi * freq_smooth * t_wave) * vol_smooth
    wave_hist = np.roll(wave_hist, -1)
    wave_hist[-1] = float(sample)

    wcy = h - 70; wamp = 48; wx0 = w//2 - WAVE_LEN//2
    ov  = frame.copy()
    cv2.rectangle(ov, (wx0-10, wcy-wamp-14), (wx0+WAVE_LEN+10, wcy+wamp+14),
                  (10,10,10), cv2.FILLED)
    frame = cv2.addWeighted(ov, 0.55, frame, 0.45, 0)
    hue  = int(np.interp(freq_smooth, [60, 1800], [130, 0]))
    cwav = tuple(int(c) for c in cv2.cvtColor(
        np.array([[[hue, 230, 230]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0])
    pts = [(wx0+i, int(wcy - wave_hist[i]*wamp)) for i in range(WAVE_LEN)]
    for i in range(1, len(pts)):
        cv2.line(frame, pts[i-1], pts[i], cwav, 2)

    # HUD
    cv2.rectangle(frame, (0,0), (w,48), (12,12,12), cv2.FILLED)
    if not right_hand and not left_hand:
        cv2.putText(frame, "Mostra tus dos manos", (w//2-140,32),
                    FONT, 0.8, (160,160,160), 2)
    else:
        try:
            midi = 12 * math.log2(max(freq_smooth,1)/440.0) + 69
            note = NOTE_NAMES[int(round(midi)) % 12]
        except: note = "?"
        cv2.putText(frame,
                    f"TONO: {int(freq_smooth)} Hz ({note})   |   VOL: {int(vol_smooth*100)}%",
                    (w//2-230, 32), FONT, 0.8, (255,255,255), 2)

    cv2.rectangle(frame, (0,52), (240,150), (12,12,12), cv2.FILLED)
    cv2.putText(frame, "MANO DERECHA = TONO",      (6,72),  FONT, 0.48, (0,220,255),   1)
    cv2.putText(frame, "  arriba=agudo abajo=grave",(6,90),  FONT, 0.43, (0,180,200),   1)
    cv2.putText(frame, "MANO IZQUIERDA = VOL",     (6,112), FONT, 0.48, (100,255,120), 1)
    cv2.putText(frame, "  separa pulgar e indice",  (6,130), FONT, 0.43, (80,200,90),   1)

    fps_c += 1
    if time.time() - fps_t >= 1.0:
        fps = fps_c; fps_c = 0; fps_t = time.time()
    cv2.putText(frame, f"FPS {fps}", (w-90,32), FONT, 0.6, (100,100,100), 1)

    cv2.imshow("Theremin - Dos Manos", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

synth.stop()
cap.release()
cv2.destroyAllWindows()
detector.close()
print("\n[FIN] ¡Hasta la próxima!")