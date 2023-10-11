import os
from glob import glob

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import tensorflow as tf

np.random.seed(42)

# preparing the dataset

METADATA_FILE = "E:\\Licenta\\HAM10000_metadata"  # the path to the dataset

tf.data.experimental.enable_debug_mode()


def load_dataset(batch_size=32, img_size=(75, 100)):
    skin_df = pd.read_csv(METADATA_FILE)

    le = LabelEncoder()  # transform the labels into numerical values
    le.fit(skin_df['dx'])
    LabelEncoder()

    skin_df['label'] = le.transform(skin_df["dx"])

    df_0 = skin_df[skin_df['label'] == 0]
    df_1 = skin_df[skin_df['label'] == 1]
    df_2 = skin_df[skin_df['label'] == 2]
    df_3 = skin_df[skin_df['label'] == 3]
    df_4 = skin_df[skin_df['label'] == 4]
    df_5 = skin_df[skin_df['label'] == 5]
    df_6 = skin_df[skin_df['label'] == 6]

    # balancing the dataset

    n_samples = 500  # the final number of images for each class
    df_0_balanced = resample(df_0, replace=True, n_samples=n_samples, random_state=42)
    df_1_balanced = resample(df_1, replace=True, n_samples=n_samples, random_state=42)
    df_2_balanced = resample(df_2, replace=True, n_samples=n_samples, random_state=42)
    df_3_balanced = resample(df_3, replace=True, n_samples=n_samples, random_state=42)
    df_4_balanced = resample(df_4, replace=True, n_samples=n_samples, random_state=42)
    df_5_balanced = resample(df_5, replace=True, n_samples=n_samples, random_state=42)
    df_6_balanced = resample(df_6, replace=True, n_samples=n_samples, random_state=42)

    skin_df_balanced = pd.concat([df_0_balanced, df_1_balanced,
                                  df_2_balanced, df_3_balanced,
                                  df_4_balanced, df_5_balanced, df_6_balanced])

    image_path = {os.path.splitext(os.path.basename(x))[0]: x
                  for x in glob(os.path.join('E:\\Dataset\\HAM10000_images', '*.jpg'))}  # associated path

    skin_df_balanced['path'] = skin_df_balanced['image_id'].map(image_path.get)

    @tf.function
    def _update_image(el):
        img = tf.io.read_file(el['path'])
        img = tf.image.decode_png(img, channels=3)
        resized_img = tf.image.resize(img, img_size)  # resize the images
        resized_img = tf.cast(resized_img, dtype=tf.float32) / 255.  # normalize to [0, 1]
        return resized_img, el['label']

    ds = tf.data.Dataset.from_tensor_slices(dict(skin_df_balanced))
    ds = ds.shuffle(skin_df_balanced.size, seed=42)  # shuffle data
    ds = ds.map(_update_image, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.cache('E:\\Dataset\\cache_dir\\cache')
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    @tf.function
    def _update_label(img, label):
        return img, tf.one_hot(label, 7)  # one hot encoded data

    ds = ds.map(_update_label, num_parallel_calls=tf.data.AUTOTUNE)

    val_split = 700
    val_ds = ds.take(val_split)
    train_ds = ds.skip(val_split)

    train_ds = train_ds.repeat().batch(batch_size, drop_remainder=True)
    val_ds = val_ds.repeat().batch(batch_size, drop_remainder=True)

    # return the tran and validation datasets
    return train_ds, val_ds
