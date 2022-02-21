import tensorflow as tf
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

#Load model test and prediction test
model = tf.keras.models.load_model("model")
model.save_weights("hwdWeights")
image = cv.imread('digit.png')[:, :, 0]
image = np.invert(np.array([image]))
prediction = model.predict(image)
print('Interpretation: {}'.format(np.argmax(prediction)))
plt.imshow(image[0])
plt.show()