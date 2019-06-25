import os
import numpy as np
import cv2
from preprocess import preprocess, oneHot

DATADIR = os.environ['HOME'] + "/Documents/Studium/HWR/new_letters/"
NPY_STORAGE = os.environ['HOME'] + "/Documents/Studium/HWR/numpy_aug/"
imHeight = 227
imWidht  = 227

nLabels = len(os.listdir(DATADIR))
init = False
trainData = np.asarray([])
trainLabels = np.asarray([]) ### todo, initcheck
testLabels = np.asarray([])
testData = np.asarray([])

labels = os.listdir(DATADIR)
nLabels = len(labels)
labelFile = open('labels.txt', 'w+')
oneHotLabels = {}
labelIdx = 0
# for label in labels:
#   labelFile.write(label + '\n')
#   #oneHotLabels[label] = oneHot(nLabels, labelIdx)
#   oneHotLabels[label] = labelIdx
#   print(labelIdx)
#   labelIdx += 1

labelFile.close()
labelIdx = -1
for label in labels:
  labelIdx += 1
  print( "Processing " + label)

  currentData = []
  currentLabels = []
  dataInitialized = False

  for x in os.walk(DATADIR + label): #1 iter
    nImgs = len(x[2])
    #currentData = np.zeros((nImgs, imHeight, imWidht, 1), dtype = 'uint8')
    currentData = []
    #currentLabels = np.zeros((nImgs, len(labels)), dtype = 'uint8')
    currentLabels = np.zeros((nImgs, 1), dtype ='uint8')
    print('shape currentLabels',np.asarray(currentLabels).shape)
    imIdx = 0
    for im in x[2]:

      #print("reading ", (DATADIR + label + '/' + im))
      image = cv2.imread(DATADIR + label + '/' + im)
      #print(currentData.shape)
      #print(image.shape)
      currentData.append(preprocess(image))
      #currentLabels[imIdx] = np.asarray(oneHotLabels[label])
      currentLabels[imIdx] = labelIdx
      imIdx += 1
      continue
      tgt = preprocess(image)
      #print("tgtdims: ", tgt.shape
      if not dataInitialized:
        #print("initadd"
        currentData = np.asarray(tgt)
        currentLabels = np.asarray(oneHotLabels[label])
        dataInitialized = True
      else:
        #print("addmore"
        currentData = np.concatenate([currentData, tgt])
        #currentData.append([tgt])
        #currentLabels.append()
        #print(oneHotLabels[label]
        #currentLabels = np.concatenate([currentLabels, oneHotLabels[label]])

     # print("curLabelShape", len(currentLabels)
      #print("curdatashape ", len(currentData)
      ##cv2.imshow('test',tgt)
      #cv2.waitKey(0)

  currentData = np.asarray(currentData)
  print(currentLabels)
  print("nData in class ", currentData.shape)
  print("check1: ", currentData.shape)
  splitIdx = int(len(currentData) * 0.75)
  splitIdx2 = int(len(currentData) * 0.90)
  train = np.asarray(currentData[:splitIdx])
  print("appending shape ", train.shape)
  trainL = np.asarray(currentLabels[:splitIdx])
  test = np.asarray(currentData[splitIdx:splitIdx2])
  testL = np.asarray(currentLabels[splitIdx:splitIdx2])
  eval = np.asarray(currentData[splitIdx2:])
  evalL = np.asarray(currentLabels[splitIdx2:])
  if init == False:
    init = True
    trainData = np.asarray(train)
    trainLabels = np.asarray(trainL)
    testData = np.asarray(test)
    testLabels = np.asarray(testL)
    evalData = np.asarray(eval)
    evalLabels = np.asarray(evalL)
  else:
    trainData = np.concatenate([trainData,train])
    trainLabels = np.concatenate([trainLabels, trainL])
    testLabels = np.concatenate([testLabels, testL])
    testData = np.concatenate([testData, test])
    evalData = np.concatenate([evalData, eval])
    evalLabels = np.concatenate([evalLabels, evalL])
print("shuffeling data...")
idx = np.random.permutation(len(trainData))
trainData, trainLabels = trainData[idx], trainLabels[idx]
idx = np.random.permutation(len(testData))
testData, testLabels = testData[idx], testLabels[idx]
print("traindatashape: ", trainData.shape)
print("storing data to folder: ", NPY_STORAGE)
np.save(NPY_STORAGE + "trainData", trainData)
np.save(NPY_STORAGE + "trainLabels", trainLabels)
np.save(NPY_STORAGE + "testData", testData)
np.save(NPY_STORAGE + "testLabels", testLabels)
np.save(NPY_STORAGE + "evalData", evalData)
np.save(NPY_STORAGE + "evalLabels", evalLabels)
