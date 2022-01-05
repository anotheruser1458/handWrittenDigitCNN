from PIL import Image
import cv2 as cv
import numpy as np
import requests

def saveImage(bytes):
    with open("hwd.png", "wb") as f:
        f.write(bytes)

def resizeImage():
    image = Image.open('hwd.png')
    image2 = image.resize((28, 28))
    image2.save("hwdSm.png")

def readImage():
    image = cv.imread("hwdSm.png")[:,:,0]
    image = np.invert(np.array([image]))
    image = image.reshape((784))
    image = ",".join([str(x) for x in image])
    return image

def sendToCloud(imageData):
    r = requests.post(
        "https://us-west1-sonny-the-robot.cloudfunctions.net/sonny_hwd",
        json={'image':imageData},
    )
    return r