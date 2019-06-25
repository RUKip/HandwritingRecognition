# from PIL import Image
# from os import listdir
# from os.path import isfile, join
import numpy as np
import cv2 as cv


class Preprocessor(object):
    # OFFSET_CROP = 50 #to have a little margin while cropping, saving contours
    BLOB_MAX_AREA = 16.0

    def binarize(self, image_path):
        print("binarizing image: ", image_path)
        image = cv.imread(image_path, 0)  # read image
        image = self.papyrusProblem(image, image_path)
        image = cv.GaussianBlur(image, (3, 3), 0)  # apply blur
        ret3, th3 = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)  # Otsu thresholding
        return th3

    # function to detect papyrus and make background lighter for otsu thresholding to work better
    def papyrusProblem(self, image, image_path):
        ret, th = cv.threshold(image, 240, 255, cv.THRESH_BINARY)
        # check how many pixels have a value over 240 (if more than 500000, its papyrus)
        num_pix = np.sum(th) / 255
        if num_pix > 500000:
            ret2, th2 = cv.threshold(image, 25, 255, cv.THRESH_BINARY)
            thInv = cv.bitwise_not(th2)
            darkPixels = np.where(thInv == 255)
            # make pixels that are very black lighter to reduce contrast and improve binarization
            image[darkPixels[0], darkPixels[1]] = image[darkPixels[0], darkPixels[1]] + 150
        return image

    # function to remove everything around the image (the letters and auxiliary items)
    # works by taking the largest connected component and removing everything around it.
    def removeExtras(self, image):

        kernelclose = cv.getStructuringElement(cv.MORPH_ELLIPSE, (50, 50))
        kernelerode = cv.getStructuringElement(cv.MORPH_ELLIPSE, (25, 25))
        # close image, to remove letters and other small items
        closed = cv.morphologyEx(image, cv.MORPH_CLOSE, kernelclose)
        nb_components, output, stats, centroids = cv.connectedComponentsWithStats(closed, connectivity=8)
        # remove background from stats
        sizes = stats[1:, -1]
        nb_components = nb_components - 1

        if nb_components > 0:
            # get maximal connected component
            max_size = max(sizes)
            # initialize mask
            mask = np.zeros((output.shape), dtype="uint8")
            # loop through connected components, until largest is found
            for i in range(0, nb_components):
                if sizes[i] >= max_size:
                    mask[output == i + 1] = 255
                    xMin = stats[i + 1, cv.CC_STAT_LEFT]
                    xMax = stats[i + 1, cv.CC_STAT_WIDTH] + xMin
                    yMin = stats[i + 1, cv.CC_STAT_TOP]
                    yMax = yMin + stats[i + 1, cv.CC_STAT_HEIGHT]

            # erode mask so that there won't be a contour around it (since we've closed it before it has become slightly larger)
            erodedmask = cv.erode(mask, kernelerode, iterations=1)
            erodedmaskI = cv.bitwise_not(erodedmask)
            # apply mask
            masked = cv.bitwise_and(image, erodedmask)
            masked = cv.bitwise_or(masked, erodedmaskI)
            # remove large stains (such as tape)
            noStains = self.removeStains(masked[yMin:yMax, xMin:xMax])
            noStainsI = cv.bitwise_not(noStains)
            # apply new mask
            final = cv.bitwise_and(noStains, masked[yMin:yMax, xMin:xMax])
            final = cv.bitwise_or(noStainsI, final)
            return final

        return image

    # function to remove large stains (such as the tape which is used to attech the papyrus/perkament)
    def removeStains(self, image):
        # use a maximum allowed size and a minimum allowed size (heuristically decided)
        MAX_SIZE = 3000
        MIN_SIZE = 20
        # compute connected components and size of background
        nb_components, output, stats, centroids = cv.connectedComponentsWithStats(cv.bitwise_not(image), connectivity=8)
        background = max(stats[:, cv.CC_STAT_AREA])
        # initialize mask
        mask = np.zeros((output.shape), dtype="uint8")
        # loop through every connected component, if not background
        for i in range(0, nb_components):
            if stats[i, cv.CC_STAT_AREA] != background:
                # if it is larger than the allowed size and the bounding box is for more
                # than 60% filled by the connected component, remove the component
                if stats[i, cv.CC_STAT_AREA] > MAX_SIZE and stats[i, cv.CC_STAT_AREA] > stats[i, cv.CC_STAT_WIDTH] * stats[i, cv.CC_STAT_WIDTH] * 0.6:
                    mask[output == i] = 255
                # if it is smaller than the allowed size, discard the connected component
                elif stats[i, cv.CC_STAT_AREA] < MIN_SIZE:
                    mask[output == i] = 255
        # mask result and return
        result = cv.bitwise_and(cv.bitwise_not(image), mask)
        return cv.bitwise_not(result)

    # make letters slightly larger, to make it easier to retrieve them
    def enlargeLetters(self, image):
        kernelopen = cv.getStructuringElement(cv.MORPH_ELLIPSE, (4, 4))
        opened = cv.morphologyEx(image, cv.MORPH_OPEN, kernelopen)
        return opened

    def despeckle(self, array):
        kerneldilate = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        dilated = cv.dilate(array, kerneldilate, iterations=2)

        contours, hierachy = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        c = min(contours, key=cv.contourArea)
        newcontours = []
        for c in contours:
            area = cv.contourArea(c)
            if (area < self.BLOB_MAX_AREA):
                newcontours.append(c)

        stencil = np.zeros(array.shape).astype(array.dtype)
        cv.drawContours(stencil, newcontours, -1, (255, 255, 0), 3)
        cv.fillPoly(stencil, [c], (255, 255, 255))
        result = cv.bitwise_or(array, stencil)
        return result


# def removeOuterborder(self, image):
#     # invert black and white
#     image = cv.bitwise_not(image)
#
#     imCopy = image.copy()
#     contours, hierachy = cv.findContours(imCopy, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#     c = max(contours, key=cv.contourArea)
#
#     # fill outside contour parts
#     stencil = np.zeros(image.shape).astype(image.dtype)
#     cv.fillPoly(stencil, [c], (255, 255, 255))
#     result = cv.bitwise_xor(image, stencil)
#
#     # invert white back to black and vice versa
#     result = cv.bitwise_not(result)
#     return result

# def arrayToImage(self, array, name):
#     numpy_array = np.array(array)
#     image = Image.fromarray(numpy_array.astype('uint8'))
#     image.save(name + ".jpg")
