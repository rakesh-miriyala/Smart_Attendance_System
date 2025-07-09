from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Perform inference on an image
results = model("test.jpg", save=True, show=True)  # Replace with your test image
