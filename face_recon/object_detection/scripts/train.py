from ultralytics import YOLO

# Load a YOLOv8 model
model = YOLO("yolov8m.pt")  # Use YOLOv8 Medium model

# Train model
model.train(data="D:/face_recon/object_detection/config.yaml", epochs=50, imgsz=640)

