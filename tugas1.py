import cv2
import numpy as np


path_file = "photo.jpg"
img = cv2.imread(filename=path_file)


def Grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)


def Threshold(img):
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return thresh


def brightening(img, alpha, beta):
    new_image = cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, beta)
    return new_image


grayscale = Grayscale(img)
threshold = Threshold(grayscale)
bright = brightening(img, alpha=1.5, beta=20)

cv2.imshow(f"{path_file} Original", img)
cv2.imshow(f"{path_file} Grayscaled", grayscale)
cv2.imshow(f"{path_file} Threshold", threshold)
cv2.imshow(f"{path_file} Brightening", bright)
cv2.waitKey()
cv2.destroyAllWindows()
