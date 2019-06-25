import cv2

from skimage.io import imread, imsave
from skimage.transform import warp, AffineTransform


def shearing(img,new_path,shearings):
    i = 0
    for shearamount in shearings:
        tform = AffineTransform(shear=shearamount)
        modified = warp(img, inverse_map=tform, mode='edge',output_shape=((50),(50)))
        imsave(new_path + str(i) + '.png', modified)
        i += 1
        tform = AffineTransform(shear=-shearamount)
        modified = warp(img, tform.inverse, mode='edge',output_shape=((50),(50)))
        imsave(new_path + str(i) + '.png', modified)
        i += 1

#img = cv2.imread("aaa.png",0)
#shearing(img,"shear",[0.1,0.2,0.3,0.4,0.5])