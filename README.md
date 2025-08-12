# Real-Time Face Recognition and Spoof Detection


## Description

This repository implements a real-time face recognition system with integrated spoof detection (liveness detection) for enhanced biometric security. It utilizes the FaceNet pre-trained model with MTCNN for face detection and TensorFlow for core operations. Additionally, it incorporates YOLO for object detection, enabling the detection of spoofing attempts using photographs or mobile devices. The model is initially trained with faces of Matthew McConaughey and Christian Bale as test subjects.

## Key Features

* **Real-time Face Recognition:** Fast and accurate facial identification.
* **Spoof Detection (Liveness Detection):** Prevents fraudulent access using object detection.
* **Object Detection:** Integrates YOLO for detecting potential spoofing objects.
* **Flexible Video Source:** Supports various video sources (webcam, RTSP, video files).
* **Fast and Accurate Results:** Optimized for performance.
* **Popular Frameworks:** Leverages TensorFlow, FaceNet, MTCNN, and YOLO.
* **Easy to Understand and Well-Structured:** Code is organized for clarity.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/a-struct-of-matter/face-recogniton]
    cd face-recogniton
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

This project combines face recognition and object detection for enhanced security.

### 1. Face Recognition Training and Detection

1.  **Preprocessing:**

    * Place images of the person to be recognized in the `image_feed` folder.
    * Run `preprocess.py` to prepare the images for training:

        ```bash
        python preprocess.py
        ```

2.  **Training:**

    * Train the face recognition model:

        ```bash
        python train.py
        ```

3.  **Detection:**

    * Run `piped.py` for combined face recognition and spoof detection.
    * Modify the `video_source` variable in `piped.py` to select your video source (e.g., `0` for webcam, RTSP link, video file path).

        ```python
        video_source = 0 # or "rtsp://..." or "video.mp4"
        ```

        ```bash
        python piped.py
        ```

### 2. Object Detection (Standalone)

1.  **Navigate to the object detection directory:**

    ```bash
    cd object_detection
    ```

2.  **Run the object detection script:**

    ```bash
    python main.py
    ```

3.  **Adjust confidence threshold:**

    * Modify the `conf` parameter in the `results=(frame, conf=0.75)` line within `main.py` to change the confidence threshold.

![WhatsApp Image 2025-08-08 at 20 03 08_e17bb56f](https://github.com/user-attachments/assets/90a899fb-8a60-4a7d-b916-5915e048a474)

![WhatsApp Image 2025-08-08 at 20 03 25_1967b191](https://github.com/user-attachments/assets/1158a0d4-028d-4dc4-9314-3fd0129fba36)

![WhatsApp Image 2025-07-30 at 22 20 02_d6ee704c](https://github.com/user-attachments/assets/dcc9c05b-7f31-4afe-b7c6-1a73d44707fc)


![WhatsApp Image 2025-07-30 at 22 20 02_960e06da](https://github.com/user-attachments/assets/293b3082-1500-44a5-9425-7a7cea3a4197)


## Important Notes:

* **Dataset:** Ensure that images are properly formatted and placed in the appropriate directory.
* **Dependencies:** Make sure all dependencies listed in `requirements.txt` are installed.
* **Video Source:** Test your video source to ensure it is working correctly.
* **Pre-trained Models:** If the pre-trained models are not included in the repo, add instructions on how to download them.


## Acknowledgments

Thanks to Face-Net for provideing pre trained models for this algorithm.
