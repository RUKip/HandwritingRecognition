import sys
import os
from pathlib import Path

import cv2 as cv

from preprocessor import Preprocessor
import segmenter
from write_unicode import write_results
from NeuralNetwork.predict import MyNetwork


def analyze_img(filename):
    """
    It will pass the image through the whole pipeline of pre-processing, segmentation, and classification.

    Args:
        filename (str): Name of the image to be processed

    Returns:
        Translated text in a list

    """
    preprocessor = Preprocessor()
    mynet = MyNetwork()
    binarized_img = preprocessor.binarize(filename)
    binarized_img = preprocessor.removeExtras(binarized_img)
    binarized_img = preprocessor.enlargeLetters(binarized_img)
    binarized_img = preprocessor.despeckle(binarized_img)
    # # To visualize binarized image
    # cv.namedWindow("test", cv.WINDOW_NORMAL)
    # cv.imshow("test", binarized_img)
    # cv.waitKey(1000)
    _, image_lines = segmenter.dilateToSegments(binarized_img, filename, save_img_with_bb=False)
    translated_text = []
    for l, line in enumerate(image_lines):
        translated_line = []
        for i, component in enumerate(line):
            print("Starting element {}/{} of line {}/{}".format(i+1, len(line), l+1, len(image_lines)))
            letter_images = segmenter.cluster_letters(component)
            for letter in letter_images:
                # # Visualize letter that will go to the NN
                # cv.namedWindow("test", cv.WINDOW_NORMAL)
                # cv.imshow("test", letter)
                # cv.waitKey(1000)
                probabilities, predicted_letter = mynet.predict_image(letter)
                print("Predicted letter: {}".format(predicted_letter))
                translated_line.append(predicted_letter)

        translated_text.append(translated_line)
        print("Translated line is: {}".format(translated_line))
    return translated_text


def main():
    # We will run your program on a Linux machine with one command line argument, namely the path to the folder
    # containing the 20 test images (see the example). So structure your program accordingly.
    # Example:
    # > python my_classifier.py mypath/testset/
    if len(sys.argv) != 2:
        print("Incorrect amount of arguments, required: 1")
        print("Example usage: python3 my_classifier.py mypath/testset/")
        sys.exit(-1)

    path_src_img = Path(sys.argv[1])  # path to test images to be classified
    if not path_src_img.is_dir():
        print("The path " + path_src_img + "is not a directory. Please set the correct path.")
        sys.exit(-1)

    results_path = Path("results-Beter-goed-gejat")  # path where we will save the resulting .txt files
    if not results_path.is_dir():
        results_path.mkdir()

    for img_name in os.listdir(path_src_img):
        img_path = path_src_img / img_name
        if img_path.stem.endswith("fused"):  # TODO: I'm guessing they will have normal and fused images..?
            print("Starting with: {}".format(img_path))
            translated_text = analyze_img(img_path.as_posix())  # They will use Linux so hardcoding this..
            results_filename = results_path / img_name
            results_filename = results_filename.with_suffix('.txt')
            write_results(translated_text, results_filename)


def testing():
    filename = "borrar/component-59.pgm"
    img = cv.imread(filename, cv.IMREAD_ANYCOLOR)
    letter_images = segmenter.cluster_letters(img)
    # letter_images = segmenter.letter_segmentation(img)
    # letter_images = segmenter.sliding_window(img)
    for i, letter in enumerate(letter_images):
        cv.imwrite("component-test-{}.png".format(i), letter)


if __name__ == "__main__":
    main()
    # testing()
