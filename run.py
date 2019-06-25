from preprocessor import Preprocessor
import segmenter
from recognizer import Recognizer
import cv2 as cv
import os

preprocessor = Preprocessor()

# preprocessor.dilateToSegments("./binarized-images/P583-Fg006-R-C01-R01-fused.jpg")

for filename in os.listdir("./image-data"):
    if filename.endswith("fused.jpg"):
        image_binarized = preprocessor.binarize("./image-data/" + filename)
        # image_binarized = preprocessor.cropImage(image_binarized)
        image_binarized = preprocessor.despeckle(image_binarized)
        image_binarized = preprocessor.removeExtras(image_binarized)

        image_binarized = preprocessor.enlargeLetters(image_binarized)

        cv.imwrite("./binarized-images/" + filename, image_binarized)  # Save test image for testing purposes
        segmenter.print_histogram("./binarized-images/" + filename)
        linesOfBoundingBoxes = segmenter.dilateToSegments(image_binarized, filename)
    else:
        continue




'''

#image_binarized = preprocessor.binarize("./image-data/P344-Fg001-R-C01-R01-fused.jpg")
image_binarized = preprocessor.binarize("./image-data/P632-Fg002-R-C01-R01-fused.jpg")
image_binarized = preprocessor.cropImage(image_binarized)
image_binarized = preprocessor.removeExtras(image_binarized)
image_binarized = preprocessor.enlargeLetters(image_binarized)
image_binarized = preprocessor.despeckle(image_binarized, 3)
#image_binarized = preprocessor.removeLargeBlobs(image_binarized)

#cv.imwrite("./binarized-images/P344-Fg001-R-C01-R01-fused.jpg", image_binarized)  # Save test image for testing purposes
cv.imwrite("./binarized-images/P632-Fg002-R-C01-R01-fused.jpg", image_binarized)  # Save test image for testing purposes

'''

# filename = "./binarized-images/P21-Fg006-R-C01-R01-fused.jpg"
#filename = "./binarized-images/P632-Fg001-R-C01-R01-fused.jpg"
#src_img = cv.imread(filename, 0)  # read image
#segmenter.print_histogram(filename)
#segmenter.print_fragment_lines(src_img, filename)
# recognizer = Recognizer()
