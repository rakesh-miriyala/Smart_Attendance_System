import os
import sys
import cv2
import time
import numpy as np
import threading
import pickle
from datetime import datetime

# ==== SETUP DJANGO ====
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'smart_attendance')))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "smart_attendance.settings")

import django
django.setup()

# ==== DJANGO IMPORTS ====
from attendanceapp.attendance_logic import mark_attendance_if_stable

# ==== ML/AI MODELS ====
from facenet_pytorch import MTCNN
from ultralytics import YOLO
from facenet import load_model, run_embeddings

# ==== MODEL LOADING ====
yolo_model = YOLO("yolov8m.pt")
face_detector = MTCNN(image_size=160, margin=20, post_process=False)
face_recognition_model = load_model("model/20180402-114759.pb")
with open("classifier.pkl", "rb") as f:
    classifier, label_encoder = pickle.load(f)

# ==== VIDEO SOURCE ====

#use 0 for webcam
rtsp_url = 0 #"rtsp://smart_attendance:123456789@192.168.0.100:554/stream2"

# ==== SHARED STATE ====
frame_lock = threading.Lock()
current_frame = None
phone_boxes = []
faces_info = []
spoof_detected = False
exit_flag = False


# ==== THREAD 1: Frame Capture ====
def frame_grabber():
    global current_frame, exit_flag
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(" Error: Unable to access video stream.")
        exit_flag = True
        return

    while not exit_flag:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                current_frame = frame.copy()
        else:
            time.sleep(0.1)
    cap.release()
    print(" Frame grabber stopped.")


# ==== THREAD 2: YOLO for Mobile Phone Detection ====
def yolo_detector():
    global current_frame, phone_boxes, exit_flag
    while not exit_flag:
        frame = None
        with frame_lock:
            if current_frame is not None:
                frame = current_frame.copy()

        if frame is not None:
            results = yolo_model(frame, agnostic_nms=True, verbose=False)
            boxes = []
            for det in results[0].boxes:
                class_id = int(det.cls)
                if class_id == 67:  # Mobile Phone
                    x1, y1, x2, y2 = det.xyxy[0].cpu().numpy().astype(int)
                    boxes.append((x1, y1, x2, y2))
            phone_boxes = boxes
        time.sleep(0.01)
    print(" YOLO detector stopped.")


# ==== THREAD 3: Face Recognition and Attendance Logic ====
def face_processor():
    global current_frame, faces_info, spoof_detected, exit_flag
    recognition_threshold = 0.78

    while not exit_flag:
        frame = None
        with frame_lock:
            if current_frame is not None:
                frame = current_frame.copy()

        if frame is not None:
            boxes, _ = face_detector.detect(frame)
            info = []
            detected_spoof = False

            if boxes is not None:
                for box in boxes:
                    x, y, x2, y2 = map(int, box)
                    face_crop = frame[max(0, y):min(frame.shape[0], y2), max(0, x):min(frame.shape[1], x2)]
                    if face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
                        continue

                    try:
                        face_resized = cv2.resize(face_crop, (160, 160))
                        gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
                        preprocessed = cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR).astype("float32") / 255.0

                        embedding = run_embeddings(preprocessed, face_recognition_model)
                        if embedding is not None and embedding.shape[0] == 512:
                            probs = classifier.predict_proba([embedding])
                            max_prob = max(probs[0])
                            emp_id = label_encoder.inverse_transform([np.argmax(probs)])[0]
                            label = emp_id if max_prob > recognition_threshold else "Unknown"

                            for px1, py1, px2, py2 in phone_boxes:
                                if x < px2 and x2 > px1 and y < py2 and y2 > py1:
                                    detected_spoof = True
                                    break

                            if label != "Unknown" and not detected_spoof:
                                mark_attendance_if_stable(emp_id, max_prob, is_spoof=False)
                                print(f" Attendance marked for {emp_id} ({max_prob:.2f})")
                            elif detected_spoof:
                                print(f" Spoof detected for {emp_id}")
                            else:
                                print(f" Unknown or low confidence")

                            info.append((x, y, x2, y2, label, max_prob))
                    except Exception as e:
                        print(f"[ERROR] Face processing: {e}")

            faces_info = info
            spoof_detected = detected_spoof
        time.sleep(0.01)
    print(" Face processor stopped.")


# ==== THREAD 4: Display Results ====
def display():
    global current_frame, faces_info, phone_boxes, spoof_detected, exit_flag
    start_time = time.time()
    frame_count = 0

    while not exit_flag:
        frame = None
        with frame_lock:
            if current_frame is not None:
                frame = current_frame.copy()

        if frame is not None:
            for x1, y1, x2, y2 in phone_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, "Phone", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            for x, y, x2, y2, label, prob in faces_info:
                color = (0, 255, 0) if label != "Unknown" else (0, 165, 255)
                text = f"{label} ({prob * 100:.1f}%)"
                cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if spoof_detected:
                cv2.putText(frame, " SPOOF DETECTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            frame_count += 1
            if (time.time() - start_time) >= 1:
                fps = frame_count / (time.time() - start_time)
                cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1] - 120, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                frame_count = 0
                start_time = time.time()

            cv2.imshow("Smart Attendance", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit_flag = True
                break
        else:
            time.sleep(0.01)

    cv2.destroyAllWindows()
    print(" Display thread stopped.")


# ==== MAIN EXECUTION ====
if __name__ == "__main__":
    threads = [
        threading.Thread(target=frame_grabber),
        threading.Thread(target=yolo_detector),
        threading.Thread(target=face_processor),
        threading.Thread(target=display)
    ]

    for t in threads:
        t.start()

    print("All threads started. Press 'q' to exit.")
    for t in threads:
        t.join()

    print(" System stopped cleanly.")
