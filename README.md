# People-tracking-and-counting-
#Tracking using machine learning in python 


#To download the applications for your OS, please visit the link: https://www.dev47apps.com/



import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

# ===== Configuration =====
DROIDCAM_URL = "http://192.168.1.2:4747/video"
WINDOW_NAME = "Real-Time People Counter"

# ===== Load YOLOv8 Model =====
model = YOLO("yolov8s.pt")

# ===== Initialize DeepSORT Tracker =====
tracker = DeepSort(max_age=30)

# ===== Initial Source: Webcam (0) =====
use_droidcam = False
unique_ids = set()
is_fullscreen = False  # Track full-screen state

def open_capture(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Failed to open source: {source}")
        return None
    return cap

def set_windowed_mode():
    screen_res = (1920, 1080)  # Replace with your screen size if different
    small_w = int(screen_res[0] * 0.30)
    small_h = int(screen_res[1] * 0.35)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, small_w, small_h)

cap = open_capture(0)  # Start with webcam
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
set_windowed_mode()

print("🟢 Press 's' to switch source, 'f' to toggle full screen, 'q' to quit.")

while True:
    if cap is None:
        break

    ret, frame = cap.read()
    if not ret or frame is None:
        print("⚠️ Failed to read frame.")
        continue

    # Run YOLO detection
    results = model(frame)[0]

    # Filter only 'person' class
    detections = []
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if int(class_id) == 0 and score > 0.4:
            detections.append(([x1, y1, x2 - x1, y2 - y1], score, 'person'))

    # Track objects
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        unique_ids.add(track_id)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show people count
    height, width, _ = frame.shape
    text = f"No. of people: {len(unique_ids)}"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0]
    text_x = width - text_size[0] - 20
    text_y = height - 20
    cv2.putText(frame, text, (text_x, text_y),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)

    # Show source info
    source_label = "DroidCam" if use_droidcam else "Webcam"
    cv2.putText(frame, f"Source: {source_label}", (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)

    # Display frame
    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        # Toggle source
        use_droidcam = not use_droidcam
        new_source = DROIDCAM_URL if use_droidcam else 0
        print(f"🔁 Switching to {'DroidCam' if use_droidcam else 'Webcam'}...")
        cap.release()
        cap = open_capture(new_source)
        unique_ids.clear()
    elif key == ord('f'):
        # Toggle full screen
        is_fullscreen = not is_fullscreen
        if is_fullscreen:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            set_windowed_mode()

cap.release()
cv2.destroyAllWindows()

