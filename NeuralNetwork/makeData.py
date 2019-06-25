from pathlib import Path
import random
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt




def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.resize_images(image, [227, 227])
    image /= 255.0  # normalize to [0,1] range
    return image

def load_and_preprocess_image(path):
    image = tf.read_file(path)
    return preprocess_image(image)

def make_dataset():
    tf.enable_eager_execution()
    tf.compat.v1.enable_eager_execution()
    tf.version.VERSION

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    keras = tf.keras

    p = Path('Characters-JPG')


    all_image_paths = list(p.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)


    label_names = sorted(item.name for item in p.glob('*/') if item.is_dir())

    label_to_index = dict((name, index) for index,name in enumerate(label_names))


    all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]


    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(load_and_preprocess_image)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))

    for label in label_ds.take(4):
      print(label_names[label.numpy()])

    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
    return image_label_ds, image_count, label_names

make_dataset()
