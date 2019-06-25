
import cv2
import numpy as np

height = 227
width = 227
nClasses = 27
def preprocess(image):
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  #print(image.shape)
  image = cv2.resize(image, (height, width))
  image = image.reshape((height, width, 1))

  image = image.astype('uint8')
  #image = image.astype('float') / 255
  # This is now done in train.py and test.py when a batch is needed. Saves RAM..
  #print(image.shape)
  return image

def oneHot(size, idx):
  vec = np.zeros((size), dtype = 'byte')
  vec[idx] = 1
  vec = vec.reshape((-1, nClasses))
  return vec
