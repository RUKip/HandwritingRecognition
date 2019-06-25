from pathlib import Path
import random
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np




def load_and_preprocess_image(path):
    im = cv2.imread(path,0)
    return im

def make_predataset2():
    #tf.enable_eager_execution()
    #tf.compat.v1.enable_eager_execution()
    #tf.version.VERSION

    #AUTOTUNE = tf.data.experimental.AUTOTUNE
    #keras = tf.keras

    #p = 'C:\handwriting\handwriting new 20 aug\HandwritingRecognition-master\pretraining\pretrainletters'
    p = 'Characters-JPG'

    all_image_names = sorted(os.listdir(p))
    all_image_paths = [str(p + '/'+  path) for path in all_image_names]
    all_images_strange_extra = [load_and_preprocess_image(path) for path in all_image_paths]

    all_image_labels = np.array([float(i) for i in range(27)])
    all_images = []
    for x in range(28):
        if x != 22:
            all_images.append(all_images_strange_extra[x])
    train_data = np.array(all_images/np.float32(255))
    return train_data, all_image_labels
