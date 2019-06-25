import numpy as np
import cv2 as cv
import os
from scipy.signal import find_peaks
from collections import namedtuple

from sklearn import cluster

CC_THRESHOLD_AREA = 100
CC_THRESHOLD_HEIGHT = 15
# class Segmenter(object):
# def segmentLetter(folder_path):
# 	print("supposed segment/binarize each letter here, from path: ", folder_path)


def fragment_lines(image):
    """
    This function finds text lines in a given binarized image. Returns the vertical position of each line's delimiters,
    as well as the center of each line
    Parameters
    ----------
    image: ndarray
        Source image where we will look for lines

    Returns
    -------
    line_separators : ndarray
        Vertical position of dividers between lines.
    peaks : ndarray
        Vertical position of histogram peaks (line positions).

    """
    projection, mean = project_image(image, HORIZONTAL_PROJECTION)
    peaks, _ = find_peaks(projection.flatten(), distance=50)

    diffs = np.diff(peaks)  # Distance between peaks
    line_separators = np.add(peaks[:-1], (diffs / 2))
    line_separators = line_separators.astype(int)

    return line_separators, peaks


def get_fragmented_lines(image):
    line_separators, peaks = fragment_lines(image)
    # TODO: this starting from top of image and ending at bottom IDK yet...
    line_separators = np.hstack((np.hstack((0, line_separators)), np.shape(image)[0]))
    fragmented_lines = []
    for i in range(np.shape(line_separators)[0] - 1):
        crop_img = image[line_separators[i]:line_separators[i + 1], :]
        fragmented_lines.append(crop_img)

    return fragmented_lines


# def droplet_line_seg(image, ):


def print_fragment_lines(image, img_filename):
    aux = img_filename.split("/")[-1]
    aux = "./segmented-lines/" + aux.split(".")[0] + "-line-"
    # Where to divide the lines
    line_separators, peaks = fragment_lines(image)
    # TODO: this starting from top of image and ending at bottom IDK yet...
    line_separators = np.hstack((np.hstack((0, line_separators)), np.shape(image)[0]))

    for i in range(np.shape(line_separators)[0] - 1):
        cropped_filename = aux + str(i) + ".jpg"
        crop_img = image[line_separators[i]:line_separators[i + 1], :]
        cv.imwrite(cropped_filename, crop_img)


VERTICAL_PROJECTION = 0
HORIZONTAL_PROJECTION = 1


def project_image(image, projection_direction):
    proj = cv.reduce(image, projection_direction, cv.REDUCE_SUM, dtype=cv.CV_32F)
    mean = np.mean(proj)
    return (proj, mean)


def print_histogram(filename, proj, peaks):
    # Create output image same height as text, 500 px wide
    m = np.max(proj)
    w = 500
    histogram = np.zeros((proj.shape[0], 500, 3))
    # Draw a line for each row
    for row in range(proj.shape[0]):
        cv.line(histogram, (0, row), (int(proj[row] * w / m), row), (255, 255, 255), 1)
    # Mark peaks
    for row in peaks:
        cv.drawMarker(histogram, (int(proj[row] * w / m), row), color=(0,0,255), markerType=cv.MARKER_TILTED_CROSS)
    cv.imwrite(filename, histogram)


def dilateToSegments(image, filename, save_img_with_bb=False):
    org_filename = filename
    if not os.path.exists('test_garbage'):
        os.mkdir("test_garbage")
    if not os.path.exists('segmentedDilation'):
        os.mkdir("segmentedDilation")
    if not os.path.exists('segmentedDilationBB'):
        os.mkdir("segmentedDilationBB")


    #
    # # erode to make all components slightly larger and more connected
    kernelerode = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    # eroded_image = cv.erode(image, kernelerode, iterations=1)

    # compute connected components and make sure not to include background
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(image - 255, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    mask = np.zeros((output.shape), dtype="uint8")

    original_image = image.copy()

    bbList = []

    original_filename = filename
    for i in range(1, nb_components):
        # only consider segments larger than the threshold height
        if stats[i, cv.CC_STAT_HEIGHT] > CC_THRESHOLD_HEIGHT:
            filename = "./segmentedDilation/" + original_filename + "_" + str(i) + ".jpg"
            xMinG = stats[i, cv.CC_STAT_LEFT]
            xMaxG = stats[i, cv.CC_STAT_WIDTH] + xMinG
            yMinG = stats[i, cv.CC_STAT_TOP]
            yMaxG = yMinG + stats[i, cv.CC_STAT_HEIGHT]

            mask[output == i] = 255
            components_image = cv.bitwise_and(original_image, original_image, mask=mask)
            line = components_image[yMinG:yMaxG, xMinG: xMaxG]

            # if the pieces are higher than average, try to split them again.
            if 4 * CC_THRESHOLD_HEIGHT < stats[i, cv.CC_STAT_HEIGHT]:
                erodedSmall = cv.dilate(image[yMinG:yMaxG, xMinG: xMaxG], kernelerode, iterations=2)
                nb_components2, output2, stats2, centroids2 = cv.connectedComponentsWithStats(erodedSmall,
                                                                                              connectivity=8)
                if nb_components2 > 2:
                    for j in range(1, nb_components2):
                        filename = "./segmentedDilation/" + original_filename \
                                   + "_" + str(i) + "_" + str(j) + ".jpg"
                        xMin = stats2[j, cv.CC_STAT_LEFT]
                        xMax = stats2[j, cv.CC_STAT_WIDTH] + xMin
                        yMin = stats2[j, cv.CC_STAT_TOP]
                        yMax = yMin + stats2[j, cv.CC_STAT_HEIGHT]

                        mask2 = np.zeros((output2.shape), dtype="uint8")
                        mask2[output2 == j] = 255
                        masked2 = cv.bitwise_and(line, line, mask=mask2)
                        line2 = line[yMin:yMax, xMin: xMax]

                        bbList = bbList + checkSegment(image[yMinG + yMin: yMax + yMinG, xMin + xMinG: xMax + xMinG ], stats2[j, cv.CC_STAT_AREA],[xMin + xMinG, xMax + xMinG, yMinG + yMin, yMax + yMinG],
                                                       filename)
                else:
                    bbList = bbList + checkSegment(image[ yMinG: yMaxG, xMinG: xMaxG], stats[i, cv.CC_STAT_AREA], [xMinG, xMaxG, yMinG, yMaxG],  filename)

            else:
                bbList = bbList + checkSegment(image[ yMinG: yMaxG, xMinG: xMaxG], stats[i, cv.CC_STAT_AREA], [xMinG, xMaxG, yMinG, yMaxG], filename)

    if save_img_with_bb:
        # To print an image with the bounding boxes over the original image
        img_with_boxes = cv.cvtColor(original_image, cv.COLOR_GRAY2BGR)
        for bb in bbList:
            cv.rectangle(img_with_boxes, (bb[0], bb[2]), (bb[1], bb[3]), (200, 200, 0), 2)
        cv.imwrite(org_filename + "-BB.png", img_with_boxes)

    return assignToLine(bbList, image)


def takeClosest(num, collection):
    return min(collection, key=lambda x: abs(x - num))

# this function checks whether the proposed segment is likely a segment of a word
# it checks the distribution of white and black(percFilled)
# it checks whether it's not too high (checkHeight)
# it checks whether the number of holes in it is not too large
def checkSegment(image, area, stats, filename):
    percFilled = area / ((stats[0] - stats[1]) * (stats[2] - stats[3]))
    if  percFilled < 0.7 and 0.0 < percFilled and checkHeight(stats) and checkNumHoles(image):

        return ([stats])

    return ([])

# checks whether a segment is not too high
def checkHeight(stats):
    if ((10* CC_THRESHOLD_HEIGHT) > (stats[3] - stats[2])) and  CC_THRESHOLD_HEIGHT < (stats[3] - stats[2]):
        return True
    else:
        return False

# check how many holes are in the image
def checkNumHoles(image):
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(image -255, connectivity=8)
    if nb_components > 6:
        return False
    else:
        return True


# not yet an ordered list
def bbListToImList(bbList, image):
    listOfIm = []
    for bb in bbList:
        listOfIm = listOfIm + [image[bb[1]:bb[3], bb[0]:bb[2]]]
    return listOfIm


def assignToLine(bbList, image):
    """
    This function assigns each connected component to the closest line,
    calculated by horizontal projection. Then it reorganizes the connected
    components of each line based on their position in the X-axis.

    Args:
        bbList (list): List of all the connected components in the image (bounding box).
        image: Original image

    Returns:
        tuple: The components ordered by line (Bounding box, Image)

    """
    # Do horizontal projection. Each peak represents a line.
    inv_img = 255-image
    proj, mean = project_image(inv_img, HORIZONTAL_PROJECTION)
    peaks, _ = find_peaks(proj.flatten(), distance=50, height=15000, prominence=500)
    # print_histogram("testing.png", proj, peaks)
    Element = namedtuple('Bounding_Box', ['Left', 'Right', 'Top', 'Bottom', 'Cx', 'Cy'])
    # Assign each component to its closest line.
    lines = [[] for _ in range(peaks.size)]
    for bounding_box in bbList:
        centerY = (bounding_box[2] + bounding_box[3]) / 2
        centerX = (bounding_box[0] + bounding_box[1]) / 2
        distances = np.absolute(peaks - centerY)
        min_dist_idx = np.argmin(distances)
        e = Element(bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3], centerX, centerY)
        lines[min_dist_idx].append(e)

    # Order the components of each line along the X-coordinate
    ordered_lines = []
    for i, line in enumerate(lines):
        centers = [i.Cx for i in line]
        centers = np.argsort(centers).astype(int)
        if len(centers) != 0:
            ordered_line = [lines[i][j] for j in centers]
            ordered_lines.append(ordered_line)

    # Crop the images with the bounding boxes
    image_lines = [[] for _ in range(len(ordered_lines))]
    for i, line in enumerate(ordered_lines):
        for element in line:
            img = image[element.Top:element.Bottom, element.Left:element.Right]
            image_lines[i].append(img)

    # # To visualize the ordered components
    # for line in image_lines:
    # 	for element in line:
    # 		cv.namedWindow("test", cv.WINDOW_NORMAL)
    # 		cv.imshow("test", element)
    # 		cv.waitKey(0)

    return ordered_lines, image_lines


def letter_segmentation(component):
    """
    We split the connected component into the letters that compose it.
    Args:
        component(numpy array): Image of the connected component

    Returns:
        List: List of the images of the letters in the component.

    """
    cv.namedWindow("test", cv.WINDOW_NORMAL)
    cv.imshow("test", component)
    cv.waitKey(0)
    kernelerode = cv.getStructuringElement(cv.MORPH_RECT, (2, 3))
    # An erosion to the black pixels, so a dilation really..
    component = cv.dilate(component, kernelerode, iterations=2)
    cv.imshow("test", component)
    cv.waitKey(0)

    proj, mean = project_image(component, VERTICAL_PROJECTION)
    proj = proj.flatten()
    # Create output image same width as text, 500 px height
    m = np.max(proj)
    h = 100
    histogram = np.zeros((h, len(proj), 3), np.uint8)

    # peaks, _ = find_peaks(proj.flatten(), distance=50, height=15000, prominence=500)
    peaks, _ = find_peaks(proj, distance=10, height=0.95*m, prominence=2000)

    # Draw a line for each column
    for col in range(component.shape[1]):
        # cv.line(histogram, (0, row), (int(proj[row] * w / m), row), (255, 255, 255), 1)
        cv.line(histogram, (col, 0), (col, int(proj[col] * h / m)), color=(255, 255, 255), thickness=1)
    # Mark peaks
    for col in peaks:
        cv.drawMarker(histogram, (col, int(proj[col] * h / m)), color=(0, 0, 255), markerType=cv.MARKER_TILTED_CROSS)

    # TODO: Crop the new images with the peaks location
    ######
    ###
    ######

    cv.imshow("test", histogram)
    cv.waitKey(0)
    # TODO: return the correct array
    return histogram


def cluster_letters(component):
    # img = cv.cvtColor(component, cv.COLOR_GRAY2BGR)
    black_pixels = np.where(component == 0)
    n_black_pixels = len(black_pixels[0])

    data_to_cluster = np.zeros((n_black_pixels, 2))
    data_to_cluster[:, 0] = black_pixels[0][:]
    data_to_cluster[:, 1] = black_pixels[1][:]

    letter_width = 40
    # K-means
    n_clusters = max(1, round(component.shape[1] / letter_width))
    kmeans = cluster.KMeans(n_clusters=n_clusters)
    kmeans.fit(data_to_cluster)
    centroids = kmeans.cluster_centers_
    ordered = np.argsort(centroids, axis=0)
    centroids = np.asarray([centroids[j][1] for j in ordered])

    # # Mixture of Gaussians
    # n_clusters = int(round(component.shape[1] / letter_width))
    # gmm = mixture.GaussianMixture(n_components=n_clusters, covariance_type='full')
    # gmm.fit(data_to_cluster)
    # centroids = gmm.means_

    # # Agglomerative clustering
    # agloclust = cluster.AgglomerativeClustering(n_clusters=None, distance_threshold=2, linkage='single')
    # agloclust.fit(data_to_cluster)
    # labels = agloclust.labels_

    # # Mean Shift
    # meanshift = cluster.MeanShift()
    # meanshift.fit(data_to_cluster)
    # centroids = meanshift.cluster_centers_

    # Create new images:
    win_size = 25  # window size that we'll use for capturing letters
    img_list = []
    for c in centroids:
        slice = component[:, int(max(0, c[1]-win_size)):int(min(c[1]+win_size, component.shape[1]))]
        img_list.append(slice)

    '''
    Code to visualize and debug
    '''
    # # Displaying for when we only have labels
    # colors = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (125, 125, 0), 4: (50, 50, 50), 5: (125, 0, 125), 6: (0, 125, 125)}
    # for i in range(n_black_pixels):
    # 	row = black_pixels[0][i]
    # 	col = black_pixels[1][i]
    # 	img[row, col] = colors[int(labels[i])]

    # for centroid in centroids:
    # 	cv.drawMarker(img, (int(round(centroid[1])), int(round(centroid[0]))), color=(0, 0, 255), markerType=cv.MARKER_TILTED_CROSS)

    return img_list


def sliding_window(component):
    cv.namedWindow("test", cv.WINDOW_NORMAL)
    cv.imshow("test", component)
    cv.waitKey(0)
    img = cv.cvtColor(component, cv.COLOR_GRAY2BGR)

    win_height = component.shape[0]
    win_width = 45

    for i in range(int(component.shape[1] / win_width)):
        top_corner = (i*win_width, 0)
        bottom_corner = (i*win_width+win_width, win_height)
        cv.rectangle(img, top_corner, bottom_corner, color=(0, 0, 255), thickness=1)

    cv.imshow("test", img)
    cv.waitKey(0)
    return img
