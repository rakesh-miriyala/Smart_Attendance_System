'''import os
import numpy as np
import tensorflow.compat.v1 as tf
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from facenet import load_model, prewhiten, load_data

tf.disable_eager_execution()

MODEL_PATH = './model/20180402-114759.pb'
DATA_DIR = './aligned_images'
CLASSIFIER_OUTPUT_PATH = 'classifier.pkl'

def get_embedding(model, images_placeholder, embeddings, phase_train_placeholder, image):
    prewhitened = prewhiten(image)
    reshaped = prewhitened.reshape(-1, 160, 160, 3)
    feed_dict = {images_placeholder: reshaped, phase_train_placeholder: False}
    embedding = model.run(embeddings, feed_dict=feed_dict)
    return embedding[0]

def main():
    print("Loading model from:", MODEL_PATH)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            load_model(MODEL_PATH)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            X = []
            y = []

            for person_name in os.listdir(DATA_DIR):
                person_path = os.path.join(DATA_DIR, person_name)
                if not os.path.isdir(person_path):
                    continue

                image_files = [f for f in os.listdir(person_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if len(image_files) == 0:
                    continue

                for img_name in image_files:
                    try:
                        img_path = os.path.join(person_path, img_name)
                        img = load_data(img_path, image_size=160)
                        emb = get_embedding(sess, images_placeholder, embeddings, phase_train_placeholder, img)
                        X.append(emb)
                        y.append(person_name)
                    except Exception as e:
                        print(f"Error processing {img_name} for {person_name}: {e}")

            if len(X) == 0:
                print("❌ No valid training data found.")
                return

            X = np.array(X)
            y = np.array(y)

            label_encoder = LabelEncoder()
            y_enc = label_encoder.fit_transform(y)

            model = SVC(kernel='linear', probability=True)
            model.fit(X, y_enc)

            with open(CLASSIFIER_OUTPUT_PATH, 'wb') as f:
                pickle.dump((model, label_encoder), f)

            print(f"\nTraining complete. Classifier saved to: {CLASSIFIER_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
'''
import os
import numpy as np
import tensorflow.compat.v1 as tf
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
# Import necessary functions from your facenet.py
from facenet import load_model, prewhiten  # prewhiten is used by run_embeddings, so it's needed here too

# For image loading and basic processing for training
import cv2  # Used for image reading and CLAHE
from PIL import Image  # Still useful for initial image loading

tf.disable_eager_execution()  # Disable eager execution for TensorFlow 1.x compatibility

# Define paths for models and data
MODEL_PATH = './model/20180402-114759.pb'  # Path to your pre-trained FaceNet model
DATA_DIR = './aligned_images'  # Directory containing the aligned faces from preprocess.py
CLASSIFIER_OUTPUT_PATH = 'classifier.pkl'  # Output path for your trained classifier


# Helper function to get embeddings from FaceNet
# This function is similar to run_embeddings in facenet.py, but for direct use in train.py
def get_embedding(sess, images_placeholder, embeddings_tensor, phase_train_placeholder, face_image):
    """
    Generates a 512-D embedding for a single face image using the FaceNet model.
    Applies prewhitening as required by FaceNet.

    Args:
        sess (tf.Session): TensorFlow session.
        images_placeholder (tf.Tensor): Placeholder for input images.
        embeddings_tensor (tf.Tensor): Tensor representing the FaceNet embeddings.
        phase_train_placeholder (tf.Tensor): Placeholder for training phase (set to False for inference).
        face_image (numpy.ndarray): The preprocessed face image (160x160x3, float32, [0,1]).

    Returns:
        numpy.ndarray: The 512-D face embedding.
    """

    # FaceNet's prewhiten function should be applied here for consistency
    # It normalizes pixel values to have zero mean and unit variance.
    # Note: run_embeddings in facenet.py already calls prewhiten, so we need to ensure
    # our preprocessing steps (CLAHE, /255.0) are applied before this function call.
    prewhitened_image = prewhiten(face_image)  # This function is from facenet.py

    # FaceNet expects a batch dimension, so reshape to (1, 160, 160, 3)
    reshaped_image = prewhitened_image.reshape(-1, 160, 160, 3)

    feed_dict = {images_placeholder: reshaped_image, phase_train_placeholder: False}
    embedding = sess.run(embeddings_tensor, feed_dict=feed_dict)
    return embedding[0]  # Return the 512-D embedding (remove the batch dimension)


def main():
    print("Starting classifier training...")
    print(f"  FaceNet model path: {MODEL_PATH}")
    print(f"  Aligned data directory: {DATA_DIR}")
    print(f"  Classifier output path: {CLASSIFIER_OUTPUT_PATH}")

    # Initialize TensorFlow graph and session
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the FaceNet model graph
            load_model(MODEL_PATH)

            # Get input and output tensors/placeholders from the loaded graph
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings_tensor = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            X = []  # List to store embeddings
            y = []  # List to store corresponding labels

            # Iterate through the aligned face dataset
            for person_name in os.listdir(DATA_DIR):
                person_path = os.path.join(DATA_DIR, person_name)
                if not os.path.isdir(person_path):
                    continue

                print(f"Processing images for: {person_name}")
                image_files = [f for f in os.listdir(person_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

                if len(image_files) == 0:
                    print(f"  No image files found for {person_name}. Skipping.")
                    continue

                for img_name in image_files:
                    try:
                        img_path = os.path.join(person_path, img_name)

                        # Load image using OpenCV for consistent preprocessing
                        img_bgr = cv2.imread(img_path)
                        if img_bgr is None:
                            print(f"  Warning: Could not read image {img_path}. Skipping.")
                            continue

                        # Ensure image is 160x160 as expected (already aligned by preprocess.py)
                        # No resizing needed here as preprocess.py already handles it.
                        # However, we need to apply the same CLAHE and normalization steps as in piped 1.py

                        # 1. Convert to Grayscale and apply CLAHE
                        gray_face = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        clahe_face = clahe.apply(gray_face)

                        # 2. Convert back to BGR (3 channels)
                        clahe_face_bgr = cv2.cvtColor(clahe_face, cv2.COLOR_GRAY2BGR)

                        # 3. Normalize pixel values to [0, 1]
                        face_processed = clahe_face_bgr.astype("float32") / 255.0

                        # Get embedding for the preprocessed face
                        emb = get_embedding(sess, images_placeholder, embeddings_tensor, phase_train_placeholder,
                                            face_processed)

                        X.append(emb)
                        y.append(person_name)
                    except Exception as e:
                        print(f"  Error processing {img_name} for {person_name}: {e}")

            if len(X) == 0:
                print("❌ No valid training data (embeddings) found. Please check DATA_DIR and preprocess images.")
                return

            X = np.array(X)
            y = np.array(y)

            print(f"\nCollected {len(X)} embeddings for {len(np.unique(y))} unique individuals.")

            # Encode string labels to numerical labels
            label_encoder = LabelEncoder()
            y_enc = label_encoder.fit_transform(y)

            # Train the SVM classifier
            print("Training SVM classifier...")
            model = SVC(kernel='linear', probability=True)  # probability=True is needed for predict_proba
            model.fit(X, y_enc)
            print("Classifier training complete.")

            # Save the trained classifier and label encoder
            with open(CLASSIFIER_OUTPUT_PATH, 'wb') as f:
                pickle.dump((model, label_encoder), f)

            print(f"\nClassifier and label encoder saved to: {CLASSIFIER_OUTPUT_PATH}")

            # Optional: Print training accuracy
            train_predictions = model.predict(X)
            accuracy = np.mean(train_predictions == y_enc)
            print(f"Training accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()