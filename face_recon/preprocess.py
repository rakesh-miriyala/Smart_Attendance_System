import os
import cv2
from mtcnn import MTCNN

def align_faces(input_dir, output_dir, image_size=160):
    detector = MTCNN()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for person_name in os.listdir(input_dir):
        person_dir = os.path.join(input_dir, person_name)
        output_person_dir = os.path.join(output_dir, person_name)

        if not os.path.isdir(person_dir):
            continue

        if not os.path.exists(output_person_dir):
            os.makedirs(output_person_dir)

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            image = cv2.imread(img_path)

            if image is None:
                print(f"Skipped: {img_path} - Unable to read.")
                continue

            # Convert BGR to RGB for MTCNN
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detections = detector.detect_faces(rgb_image)

            if not detections:
                print(f"No face detected in {img_path}")
                continue

            # Pick the most confident face
            det = max(detections, key=lambda d: d['confidence'])
            if det['confidence'] < 0.90:
                print(f"Low confidence face in {img_path}, skipping.")
                continue

            x, y, w, h = det['box']
            x, y = max(0, x), max(0, y)
            face = image[y:y + h, x:x + w]
            try:
                face_resized = cv2.resize(face, (image_size, image_size))
                save_path = os.path.join(output_person_dir, img_name)
                cv2.imwrite(save_path, face_resized)
                print(f"Saved aligned face: {save_path}")
            except Exception as e:
                print(f"Failed to process {img_path}: {e}")

if __name__ == "__main__":
    align_faces("./image_feed", "./aligned_images")
