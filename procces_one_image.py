import os
from glob import glob

import numpy as np
import pandas as pd
import tensorflow_addons as tfa

np.random.seed(42)
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

METADATA_FILE = "E:\\Licenta\\HAM10000_metadata"


# function that handle the image loaded: reads, resizes and normalizes
def process_single_image(image_path, img_size=(75, 100)):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    resized_img = tf.image.resize(img, img_size)
    resized_img = tf.cast(resized_img, dtype=tf.float32) / 255.

    return resized_img


def load_single_image(image_path):
    skin_df = pd.read_csv(METADATA_FILE)

    le = LabelEncoder()
    le.fit(skin_df['dx'])
    LabelEncoder()

    skin_df['label'] = le.transform(skin_df["dx"])

    image_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                       for x in glob(os.path.join('E:\\Dataset\\HAM10000_images', '*.jpg'))}

    skin_df['path'] = skin_df['image_id'].map(image_path_dict.get)

    print(f"Image Path: {image_path}")
    print(f"Available Image Paths: {skin_df['path']}")

    image_info = skin_df[skin_df['path'] == image_path]
    label = image_info['label']

    img = process_single_image(image_path)

    return img, label

# load the trained model
def load_model():
    model = tf.keras.models.load_model('models/trained_model.h5',
                                       custom_objects={'gelu': tfa.activations.gelu})

    return model

# get the class labels
class_labels = ['Actinic keratoses and intraepithelial carcinoma / Bowen\'s disease',
                'basal cell carcinoma',
                'benign keratosis',
                'dermatofibroma',
                'melanoma',
                'melanocytic nevi',
                'vascular lesions']

def process_and_predict(image_path):
    img = process_single_image(image_path)

    image_batch = img[tf.newaxis, ...]
    # load the pre-trained model
    model = load_model()

    # make predictions using the model
    predictions = model.predict(image_batch)

    predicted_label = class_labels[np.argmax(predictions)]

    prediction_result = {
        'predicted_label': predicted_label,
    }

    return prediction_result

