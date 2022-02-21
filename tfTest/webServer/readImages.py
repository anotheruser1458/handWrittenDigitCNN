import cv2 as cv
import numpy as np

image = cv.imread("digit2.png")[:, :, 0]
image = np.invert(np.array([image]))
image = image.reshape((784))
image = ",".join([str(x) for x in image])
print(image)