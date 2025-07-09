import tensorflow as tf
import numpy as np
from PIL import Image
import os

def load_model(model_path):
    print("Loading model from:", model_path)
    tf.compat.v1.disable_eager_execution()
    sess = tf.compat.v1.Session()
    model_exp = os.path.expanduser(model_path)
    with tf.compat.v1.gfile.GFile(model_exp, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    print("Model loaded successfully.")
    return sess

def run_embeddings(face_image, sess):
    try:
        images_placeholder = sess.graph.get_tensor_by_name("input:0")
        phase_train_placeholder = sess.graph.get_tensor_by_name("phase_train:0")
        embeddings_tensor = sess.graph.get_tensor_by_name("embeddings:0")
        face_image = prewhiten(face_image)
        face_image = np.expand_dims(face_image, axis=0)
        feed_dict = {images_placeholder: face_image, phase_train_placeholder: False}
        embeddings = sess.run(embeddings_tensor, feed_dict=feed_dict)
        return embeddings[0]
    except Exception as e:
        print(f"Error in generating embeddings: {e}")
        return None

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = (x - mean) / std_adj
    return y

def load_data(image_path, do_crop=False, do_flip=False, image_size=160):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((image_size, image_size))
    img = np.asarray(img, dtype=np.float32)
    return img
