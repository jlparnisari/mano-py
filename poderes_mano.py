"""
=============================================================
  ⚡ PODERES DE MANO
  Uso: python3 poderes_mano.py

  PUÑO CERRADO  → cargás el poder
  MANO ABIERTA  → lanzás el poder
  TECLAS:
    F  → Fuego
    R  → Rayo
    H  → Hielo
    V  → Viento
    Q  → Salir
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
import random

# ─── Audio con afplay ─────────────────────────────────────────────────────────
SAMPLE_RATE = 22050

def play_sound_async(freq_start, freq_end, duration, vol=0.6, wave_type="sine"):
    def _play():
        n      = int(SAMPLE_RATE * duration)
        t      = np.linspace(0, duration, n, dtype=np.float64)
        freqs  = np.linspace(freq_start, freq_end, n)
        phase  = np.cumsum(2 * math.pi * freqs / SAMPLE_RATE)

        if wave_type == "sine":
            w = np.sin(phase)
        elif wave_type == "saw":
            w = 2 * (phase / (2*math.pi) % 1) - 1
        elif wave_type == "noise":
            w = np.random.uniform(-1, 1, n)
            # filtrar con convolución para suavizar
            kernel = np.ones(50) / 50
            w = np.convolve(w, kernel, mode='same')
        else:
            w = np.sin(phase)

        # Envelope: attack + decay
        env = np.ones(n)
        att = int(n * 0.05)
        dec = int(n * 0.1)
        env[:att] = np.linspace(0, 1, att)
        env[-dec:] = np.linspace(1, 0, dec)
        w *= env

        samples = (w * vol * 28000).astype(np.int16)
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        with wave.open(tmp.name, 'wb') as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
            wf.writeframes(samples.tobytes())
        try:
            subprocess.run(['afplay', tmp.name], capture_output=True)
        finally:
            try: os.unlink(tmp.name)
            except: pass

    threading.Thread(target=_play, daemon=True).start()

def sound_fuego():   play_sound_async(200, 80,  0.6, 0.5, "noise")
def sound_rayo():    play_sound_async(800, 2000, 0.3, 0.7, "saw")
def sound_hielo():   play_sound_async(1200, 400, 0.8, 0.4, "sine")
def sound_viento():  play_sound_async(300, 600, 0.7, 0.3, "noise")
def sound_carga():   play_sound_async(300, 600, 0.4, 0.3, "sine")

# ─── Partículas ───────────────────────────────────────────────────────────────
class Particle:
    def __init__(self, x, y, poder):
        self.x    = float(x)
        self.y    = float(y)
        angle     = random.uniform(0, 2*math.pi)
        speed     = random.uniform(3, 12)
        self.vx   = math.cos(angle) * speed
        self.vy   = math.sin(angle) * speed - random.uniform(0, 3)
        self.life = 1.0
        self.decay= random.uniform(0.02, 0.06)
        self.size = random.randint(4, 16)
        self.poder= poder

    def update(self):
        self.x    += self.vx
        self.y    += self.vy
        self.life -= self.decay
        if self.poder == "fuego":
            self.vy   -= 0.3   # sube
            self.size  = max(1, self.size - 0.3)
        elif self.poder == "hielo":
            self.vx   *= 0.92  # frena
            self.vy   *= 0.92
        elif self.poder == "rayo":
            self.vx   += random.uniform(-1, 1)  # zigzag
            self.vy   += random.uniform(-1, 1)
        elif self.poder == "viento":
            self.vx   *= 1.05  # acelera
        return self.life > 0

    def draw(self, frame):
        a = max(0.0, min(1.0, self.life))
        x, y = int(self.x), int(self.y)
        if x < 0 or y < 0 or x >= frame.shape[1] or y >= frame.shape[0]:
            return
        if self.poder == "fuego":
            r = int(255)
            g = int(np.interp(self.life, [0,1], [0, 180]))
            b = 0
            color = (b, g, r)
        elif self.poder == "hielo":
            r = int(np.interp(self.life, [0,1], [100, 200]))
            g = int(np.interp(self.life, [0,1], [200, 240]))
            b = 255
            color = (b, g, r)
        elif self.poder == "rayo":
            color = (int(50*a), int(50*a), int(255*a))
        elif self.poder == "viento":
            g = int(200 * a)
            color = (g, 255, g)
        else:
            color = (200, 200, 200)

        cv2.circle(frame, (x, y), max(1, int(self.size * a)), color, cv2.FILLED)

class RayoSegment:
    """Segmento de rayo zigzag entre dos puntos."""
    def __init__(self, x1, y1, x2, y2):
        self.pts  = self._make(x1, y1, x2, y2)
        self.life = 1.0

    def _make(self, x1, y1, x2, y2):
        pts = [(x1, y1)]
        steps = 12
        for i in range(1, steps):
            t  = i / steps
            mx = x1 + (x2-x1)*t + random.randint(-30, 30)
            my = y1 + (y2-y1)*t + random.randint(-30, 30)
            pts.append((int(mx), int(my)))
        pts.append((x2, y2))
        return pts

    def update(self):
        self.life -= 0.15
        return self.life > 0

    def draw(self, frame):
        color = (int(100*self.life), int(100*self.life), 255)
        thick = max(1, int(3 * self.life))
        for i in range(1, len(self.pts)):
            cv2.line(frame, self.pts[i-1], self.pts[i], color, thick)

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
    num_hands=1,
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

# ─── Estado del juego ─────────────────────────────────────────────────────────
PODERES = ["fuego", "rayo", "hielo", "viento"]
TECLAS  = {"f": "fuego", "r": "rayo", "h": "hielo", "v": "viento"}
NOMBRES = {"fuego": "FUEGO", "rayo": "RAYO", "hielo": "HIELO", "viento": "VIENTO"}
COLORES_HUD = {
    "fuego":  (0, 100, 255),
    "rayo":   (0, 220, 255),
    "hielo":  (255, 220, 100),
    "viento": (100, 255, 100),
}

poder_actual  = "fuego"
carga         = 0.0        # 0.0 a 1.0
cargando      = False
disparado     = False
ultimo_estado = "abierta"  # "cerrada" o "abierta"
particles     = []
rayos         = []
timestamp     = 0

FONT = cv2.FONT_HERSHEY_SIMPLEX

def es_mano_cerrada(lm, w, h):
    """Detecta si la mano está cerrada (puño)."""
    # Punta de cada dedo vs su nudillo base
    dedos = [
        (8,  5),   # índice
        (12, 9),   # medio
        (16, 13),  # anular
        (20, 17),  # meñique
    ]
    cerrados = 0
    for tip, base in dedos:
        if lm[tip].y > lm[base].y:  # punta más abajo que base = cerrado
            cerrados += 1
    return cerrados >= 3

def centro_mano(lm, w, h):
    xs = [lm[i].x * w for i in range(21)]
    ys = [lm[i].y * h for i in range(21)]
    return int(np.mean(xs)), int(np.mean(ys))

def punta_dedo_medio(lm, w, h):
    return int(lm[12].x * w), int(lm[12].y * h)

def lanzar_poder(cx, cy, poder):
    n = {"fuego": 60, "rayo": 30, "hielo": 50, "viento": 70}[poder]
    for _ in range(n):
        particles.append(Particle(cx, cy, poder))
    if poder == "rayo":
        # Rayos en varias direcciones
        for _ in range(5):
            dx = random.randint(-200, 200)
            dy = random.randint(-200, 200)
            rayos.append(RayoSegment(cx, cy, cx+dx, cy+dy))
    sounds = {"fuego": sound_fuego, "rayo": sound_rayo,
              "hielo": sound_hielo, "viento": sound_viento}
    sounds[poder]()

print("\n[LISTO] Apuntá tu mano a la cámara.")
print("  PUÑO CERRADO → cargás el poder")
print("  MANO ABIERTA → lanzás el poder")
print("  F=Fuego  R=Rayo  H=Hielo  V=Viento  Q=Salir\n")

fps_t = time.time(); fps_c = 0; fps = 0

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    # Detectar
    small  = cv2.resize(frame, (480, 270))
    rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp += 1
    detector.detect_async(mp_img, timestamp)

    with data_lock:
        lm_list = (hands_data.get("Left") or hands_data.get("Right"))

    cx, cy = w//2, h//2

    if lm_list:
        cx, cy = centro_mano(lm_list, w, h)
        cerrada = es_mano_cerrada(lm_list, w, h)

        if cerrada:
            # Cargando
            if not cargando:
                cargando = True
                sound_carga()
            carga = min(1.0, carga + 0.025)
            disparado = False

            # Efecto de carga: partículas orbitando
            if random.random() < carga:
                angle  = random.uniform(0, 2*math.pi)
                radius = int(np.interp(carga, [0,1], [60, 20]))
                px = cx + int(math.cos(angle) * radius)
                py = cy + int(math.sin(angle) * radius)
                p  = Particle(px, py, poder_actual)
                p.vx *= 0.3; p.vy *= 0.3
                particles.append(p)

            # Dibujar anillo de carga
            radio_carga = int(np.interp(carga, [0,1], [40, 80]))
            color_carga = COLORES_HUD[poder_actual]
            cv2.circle(frame, (cx, cy), radio_carga, color_carga, 3)
            # Arco de progreso
            angulo_fin = int(360 * carga)
            cv2.ellipse(frame, (cx, cy), (radio_carga+10, radio_carga+10),
                        -90, 0, angulo_fin, color_carga, 4)

        else:
            cargando = False
            # Si venía cargado → LANZAR
            if carga > 0.2 and not disparado:
                disparado = True
                lanzar_poder(cx, cy, poder_actual)
                carga = 0.0
            elif carga > 0:
                carga = max(0.0, carga - 0.05)  # descarga lenta

        # Dibujar esqueleto mano
        CONNECTIONS = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
            (5,9),(9,13),(13,17),
        ]
        col_hueso = COLORES_HUD[poder_actual]
        for a, b in CONNECTIONS:
            cv2.line(frame,
                     (int(lm_list[a].x*w), int(lm_list[a].y*h)),
                     (int(lm_list[b].x*w), int(lm_list[b].y*h)),
                     col_hueso, 1)
        for i in range(21):
            cv2.circle(frame, (int(lm_list[i].x*w), int(lm_list[i].y*h)),
                       4, col_hueso, cv2.FILLED)

    # ── Actualizar y dibujar partículas ───────────────────────────────────────
    particles = [p for p in particles if p.update()]
    for p in particles:
        p.draw(frame)

    rayos = [r for r in rayos if r.update()]
    for r in rayos:
        r.draw(frame)

    # ── HUD ───────────────────────────────────────────────────────────────────
    # Barra de carga
    bw = 200; bh = 18
    bx = w//2 - bw//2; by = h - 50
    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (40,40,40), cv2.FILLED)
    fill = int(bw * carga)
    if fill > 0:
        cv2.rectangle(frame, (bx, by), (bx+fill, by+bh),
                      COLORES_HUD[poder_actual], cv2.FILLED)
    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (120,120,120), 1)
    cv2.putText(frame, "CARGA", (bx, by-6), FONT, 0.45, (180,180,180), 1)

    # Panel superior
    cv2.rectangle(frame, (0,0), (w, 52), (15,15,15), cv2.FILLED)
    col = COLORES_HUD[poder_actual]

    # Poder actual
    cv2.putText(frame, NOMBRES[poder_actual],
                (w//2 - 60, 35), FONT, 1.0, col, 2)

    # Iconos de poderes
    iconos = [("F", "fuego"), ("R", "rayo"), ("H", "hielo"), ("V", "viento")]
    for idx, (tecla, p) in enumerate(iconos):
        ix = 20 + idx * 70
        iy = 30
        activo = (p == poder_actual)
        bg = COLORES_HUD[p] if activo else (50,50,50)
        cv2.rectangle(frame, (ix-5, iy-22), (ix+55, iy+8), bg, cv2.FILLED)
        cv2.putText(frame, f"[{tecla}] {p[:3].upper()}",
                    (ix, iy), FONT, 0.42,
                    (0,0,0) if activo else (150,150,150), 1)

    # Instrucción
    if not lm_list:
        cv2.putText(frame, "Mostra tu mano", (w//2-100, h//2),
                    FONT, 1.0, (180,180,180), 2)
    elif carga > 0.15:
        cv2.putText(frame, "ABRE LA MANO PARA LANZAR",
                    (w//2-200, h-70), FONT, 0.65, col, 2)
    else:
        cv2.putText(frame, "CIERRA EL PUNO PARA CARGAR",
                    (w//2-210, h-70), FONT, 0.65, (160,160,160), 1)

    fps_c += 1
    if time.time() - fps_t >= 1.0:
        fps = fps_c; fps_c = 0; fps_t = time.time()
    cv2.putText(frame, f"FPS {fps}", (w-90, 35), FONT, 0.6, (100,100,100), 1)

    cv2.imshow("Poderes de Mano", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    for tecla, poder in TECLAS.items():
        if key == ord(tecla):
            poder_actual = poder
            carga = 0.0
            print(f"[PODER] {NOMBRES[poder]}")

cap.release()
cv2.destroyAllWindows()
detector.close()
print("\n[FIN] ¡Hasta la próxima!")