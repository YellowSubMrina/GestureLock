import cv2
import mediapipe as mp
import time
import numpy as np
from collections import deque
from pynput.mouse import Button, Controller
from screeninfo import get_monitors
import insightface
from numpy.linalg import norm

mouse = Controller()
monitors = get_monitors()
wScr, hScr = monitors[0].width, monitors[0].height

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils
DrawingSpec = mp.solutions.drawing_utils.DrawingSpec
GREEN_COLOR = (0, 255, 0)
RED_COLOR = (0, 0, 255)

class GestureStabilizer:
    def __init__(self):
        self.buffer_size = 10
        self.min_active_duration = 0.1
        self.last_state_change = 0
        self.current_state = False
        self.activation_threshold = 0.1
        self.deactivation_threshold = 0.1
        self.detection_buffer = deque(maxlen=self.buffer_size)

    def update_state(self, current_detection):
        self.detection_buffer.append(1 if current_detection else 0)
        positive_ratio = sum(self.detection_buffer) / len(self.detection_buffer)
        current_time = time.time()

        if self.current_state:
            if positive_ratio < self.deactivation_threshold:
                if (current_time - self.last_state_change) > self.min_active_duration:
                    self.current_state = False
                    self.last_state_change = current_time
        else:
            if positive_ratio > self.activation_threshold:
                self.current_state = True
                self.last_state_change = current_time

        return self.current_state

def map_coordinates(x, y, cam_w, cam_h):
    screen_x = int(x * wScr / cam_w)
    screen_y = int(y * hScr / cam_h)
    return screen_x, screen_y

def is_touch_index_thumb(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_index_distance = ((thumb_tip.x - index_tip.x) ** 2 +
                             (thumb_tip.y - index_tip.y) ** 2) ** 0.5
    return thumb_index_distance <= 0.04

face_model = insightface.app.FaceAnalysis(name="buffalo_l")
face_model.prepare(ctx_id=0)

my_face_embedding = np.load("my_face_embedding.npy")

def is_my_face(frame, model, my_embedding, threshold=0.4):
    faces = model.get(frame)
    if not faces:
        return False
    embedding = faces[0].normed_embedding
    distance = 1 - np.dot(embedding, my_embedding)
    return distance < threshold

cap = cv2.VideoCapture(0)

stabilizer = GestureStabilizer()
prev_state = None
color = RED_COLOR
window_size = 5
mouse_prev_xs = deque(maxlen=5)
mouse_prev_ys = deque(maxlen=5)
authorized = False
last_face_check_time = 0
face_check_interval = 60

print("Запуск программы...")

while True:
    success, img = cap.read()
    if not success:
        continue

    h, w, c = img.shape

    # Проверка наличия лица каждые несколько секунд
    current_time = time.time()
    if current_time - last_face_check_time > face_check_interval:
        authorized = is_my_face(img, face_model, my_face_embedding)
        last_face_check_time = current_time

    if not authorized:
        cv2.putText(img, "Face Not Authorized", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, RED_COLOR, 2)
        cv2.imshow("Hand Controlled Mouse", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
            x, y = landmarks[8][0], landmarks[8][1]

            screen_x, screen_y = wScr * x, hScr * y

            mouse_prev_xs.append(screen_x)
            mouse_prev_ys.append(screen_y)
            screen_x = sorted(mouse_prev_xs)[len(mouse_prev_xs) // 2]
            screen_y = sorted(mouse_prev_ys)[len(mouse_prev_ys) // 2]

            mouse.position = (wScr - screen_x, screen_y)

            is_touch = is_touch_index_thumb(hand_landmarks)
            stab_touch = stabilizer.update_state(is_touch)

            if not prev_state and stab_touch:
                mouse.press(Button.left)
                color = GREEN_COLOR
            elif prev_state and not stab_touch:
                mouse.release(Button.left)
                color = RED_COLOR

            prev_state = stab_touch

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS, DrawingSpec(color=color))

    radius = 10
    offset = 20
    y_pos = 40

    any_face_color = GREEN_COLOR if face_model.get(img) else RED_COLOR
    cv2.circle(img, (offset, y_pos), radius, any_face_color, -1)

    authorized_color = GREEN_COLOR if authorized else RED_COLOR
    cv2.circle(img, (offset + 2 * radius + 10, y_pos), radius, authorized_color, -1)

    cv2.imshow("Hand Controlled Mouse", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(0.005)

cap.release()
cv2.destroyAllWindows()
