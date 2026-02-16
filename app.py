import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import pythoncom
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# ---------------- PAGE CONFIG ----------------
st.set_page_config(layout="wide")
st.title("✋ Hand Gesture Volume Control")

# ---------------- SESSION STATE ----------------
defaults = {
    "run": False,
    "hands": 0,
    "volume": 0,
    "fps": 0,
    "muted": False
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------- SIDEBAR ----------------
st.sidebar.header("Camera Controls")

if st.sidebar.button("Start Camera"):
    st.session_state.run = True

if st.sidebar.button("Stop Camera"):
    st.session_state.run = False

capture = st.sidebar.button("Capture Frame")

st.sidebar.header("Detection Parameters")
det_conf = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.7, 0.05)
track_conf = st.sidebar.slider("Tracking Confidence", 0.0, 1.0, 0.7, 0.05)
max_hands = st.sidebar.slider("Max Hands", 1, 4, 2)

status_box = st.sidebar.empty()
status_panel = st.sidebar.empty()

# ---------------- AUDIO SETUP ----------------
pythoncom.CoInitialize()

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume_ctrl = cast(interface, POINTER(IAudioEndpointVolume))
min_vol, max_vol = volume_ctrl.GetVolumeRange()[:2]

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
frame_placeholder = st.empty()

# ---------------- GESTURE FUNCTION ----------------
def is_open_palm(lm_list):
    finger_tips = [4, 8, 12, 16, 20]
    finger_knuckles = [2, 6, 10, 14, 18]
    return all(lm_list[t][1] < lm_list[k][1] for t, k in zip(finger_tips, finger_knuckles))

# ---------------- CAMERA LOOP ----------------
if st.session_state.run:

    cap = cv2.VideoCapture(0)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_hands,
        min_detection_confidence=det_conf,
        min_tracking_confidence=track_conf
    )

    status_box.success("Camera Active")

    prev_time = 0

    # ---- stable mute variables ----
    last_mute_time = 0
    MUTE_COOLDOWN = 1.5
    palm_was_open = False

    while st.session_state.run:

        ret, img = cap.read()
        if not ret:
            st.warning("Camera not found")
            break

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # -------- HAND COUNT --------
        hand_count = 0
        if results.multi_hand_landmarks:
            hand_count = len(results.multi_hand_landmarks)

        st.session_state.hands = hand_count

        # -------- PROCESS HANDS --------
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                lm_list = []
                h, w, _ = img.shape

                for lm in hand_landmarks.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append((cx, cy))

                if lm_list:
                    x1, y1 = lm_list[4]   # thumb
                    x2, y2 = lm_list[8]   # index
                    distance = np.hypot(x2 - x1, y2 - y1)

                    # -------- VOLUME CONTROL --------
                    vol = np.interp(distance, [30, 200], [min_vol, max_vol])
                    volume_ctrl.SetMasterVolumeLevel(vol, None)
                    st.session_state.volume = int(np.interp(distance, [30, 200], [0, 100]))

                    # draw pinch line
                    cv2.circle(img, (x1, y1), 10, (255, 0, 0), -1)
                    cv2.circle(img, (x2, y2), 10, (255, 0, 0), -1)
                    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    # -------- STABLE MUTE --------
                    current_time = time.time()
                    palm_open = is_open_palm(lm_list)

                    if palm_open and not palm_was_open:
                        if current_time - last_mute_time > MUTE_COOLDOWN:
                            st.session_state.muted = not st.session_state.muted
                            volume_ctrl.SetMute(st.session_state.muted, None)
                            last_mute_time = current_time

                    palm_was_open = palm_open

                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # -------- FPS --------
        curr_time = time.time()
        st.session_state.fps = int(1 / (curr_time - prev_time + 0.0001))
        prev_time = curr_time

        # -------- UI UPDATE --------
        status_panel.markdown(f"""
### Detection Status
**Hands Detected:** {st.session_state.hands}  
**Volume:** {st.session_state.volume}%  
**Muted:** {st.session_state.muted}  
**FPS:** {st.session_state.fps}  
**Resolution:** {img.shape[1]}x{img.shape[0]}
""")

        frame_placeholder.image(img, channels="BGR")

        # -------- CAPTURE --------
        if capture:
            cv2.imwrite("captured_frame.jpg", img)
            st.sidebar.success("Frame Saved")

    cap.release()
    status_box.error("Camera Stopped")
