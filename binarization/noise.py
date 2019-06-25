import numpy as np
import cv2


def noise(img,new_path,noiserates):    
    i = 0      
    mean = 0
    row,col= img.shape
    for noiserate in noiserates:
        var = noiserate
        sigma = var**5
        gauss = np.random.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)
        noiseimg = img + gauss
        p = new_path + str(i) + '.png'
        cv2.imwrite(p,noiseimg)
        i += 1

