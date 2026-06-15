from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import numpy as np
import imutils
import dlib
import cv2

# =======================
# AUDIO
# =======================
mixer.init()
mixer.music.load("D:\Codink\pythonProject1\Driver Drowsiness Detection\music.wav")

# =======================
# FUNCTIONS
# =======================
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

def get_face_direction(shape):
    nose_tip = shape[30]
    nose_bridge = shape[27]

    dx = nose_tip[0] - nose_bridge[0]
    dy = nose_tip[1] - nose_bridge[1]

    face_left = shape[0][0]
    face_right = shape[16][0]
    face_width = face_right - face_left

    normalized_dx = dx / face_width
    normalized_dy = dy / face_width

    if normalized_dy > 0.30:
        return "bawah"
    elif normalized_dx < -0.05:
        return "kiri"
    elif normalized_dx > 0.05:
        return "kanan"
    else:
        return "depan"

# =======================
# CONSTANTS
# =======================
EYE_THRESH = 0.25
EYE_FRAME_CHECK = 20

MOUTH_THRESH = 0.65
MOUTH_FRAME_CHECK = 35

DIRECTION_FRAME_CHECK = 40

# =======================
# DLIB INITIALIZATION
# =======================
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("D:\Codink\pythonProject1\Driver Drowsiness Detection\shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

# =======================
# VIDEO CAPTURE
# =======================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

eye_flag = 0
yawn_flag = 0
direction_flag = 0

# =======================
# MAIN LOOP
# =======================
while True:
    ret, frame = cap.read()

    # 🛑 GUARD: frame wajib valid
    if not ret or frame is None:
        print("[WARNING] Frame tidak terbaca")
        continue

    frame = imutils.resize(frame, width=450)

    # ✅ RGB VERSION (AMAN UNTUK DLIB)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)

    # 🔥 DETEKSI WAJAH
    subjects = detect(rgb)

    for subject in subjects:
        shape = predict(rgb, subject)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]

        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
        mar = mouth_aspect_ratio(mouth)
        face_dir = get_face_direction(shape)

        # DRAW CONTOURS
        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (255, 255, 0), 1)

        nose = shape[27:36]
        cv2.drawContours(frame, [cv2.convexHull(nose)], -1, (0, 255, 255), 1)

        # =======================
        # DETECTION LOGIC
        # =======================
        if ear < EYE_THRESH:
            eye_flag += 1
            if eye_flag >= EYE_FRAME_CHECK:
                cv2.putText(frame, "KANTUK TERDETEKSI!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                mixer.music.play()
        else:
            eye_flag = 0

        if mar > MOUTH_THRESH:
            yawn_flag += 1
            if yawn_flag >= MOUTH_FRAME_CHECK:
                cv2.putText(frame, "ANDA MENGUAP!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                mixer.music.play()
        else:
            yawn_flag = 0

        if face_dir != "depan":
            direction_flag += 1
            if direction_flag >= DIRECTION_FRAME_CHECK:
                cv2.putText(frame, "PERHATIKAN ARAH DEPAN!", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                mixer.music.play()
        else:
            direction_flag = 0

        # =======================
        # DISPLAY INFO
        # =======================
        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Arah: {face_dir}", (300, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
