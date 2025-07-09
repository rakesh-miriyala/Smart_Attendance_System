import cv2
from ultralytics import YOLO

# Load YOLOv8 model (pretrained on COCO dataset)
model = YOLO("yolov8m.pt")

# Open webcam (change 0 to a file path for video)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Extract detections only for "cell phone" (Class ID 67 in COCO dataset)
    filtered_detections = [det for det in results[0] if det.boxes.cls.tolist()[0] == 67]

    # Draw only mobile phone detections
    for det in filtered_detections:
        frame = det.plot()

    # Display the frame
    cv2.imshow("YOLOv8 Mobile Phone Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
