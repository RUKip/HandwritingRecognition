import cv2


def rotation(img,new_path,rotationangles):
    i = 0     
    rows = img.shape[0]
    cols = img.shape[1]
    img_center = (cols / 2, rows / 2)     
    for angle in rotationangles:
        M = cv2.getRotationMatrix2D(img_center, angle, 1)
        M2 = cv2.getRotationMatrix2D(img_center, 360 - angle, 1)
        rotated_image = cv2.warpAffine(img, M, (cols, rows), borderValue=(255,255,255))
        rotated_image2 = cv2.warpAffine(img, M2, (cols, rows), borderValue=(255,255,255))
        p = new_path + str(i) + '.jpg'
        cv2.imwrite(p,rotated_image)
        i += 1
        p = new_path + str(i) + '.jpg'
        cv2.imwrite(p,rotated_image2)
        i+= 1

