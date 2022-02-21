# handWrittenDigitCNN
Neural network trained to recognize hand written digits.

## Overview
This is a personal project which involved training a TensorFlow neural network to recognize hand written digits, deploying the model to the cloud, and creating a frontend user interface which allows people to draw their own digits on screen and recieve the model's prediction of what digit it thinks it sees. 

## Model Training
The tfTest folder contains all the code related to training and testing the model. Below are the steps I took to ensure the model worked properly before deploying it to the cloud.
### Training the Model
'''python
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
'''
## Deploy to Google Cloud Platform Function

## Web Server

### Frontend
### Backend

## Deploy to Heroku

## Summary
