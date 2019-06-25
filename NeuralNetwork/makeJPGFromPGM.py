from pathlib import Path
import random
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

tf.enable_eager_execution()

load_path = Path('HWR/Characters')
save_path = Path('HWR/Characters-JPG')

all_image_paths = list(load_path.glob('*/*'))
all_image_paths = [str(path)[14:] for path in all_image_paths]

print(all_image_paths[0])

for path in all_image_paths:
    image = Image.open(str(load_path) + str(path))
    image.save(str(save_path) + str(path)[:-3] +'jpg')
