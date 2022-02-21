# handWrittenDigitCNN
Neural network trained to recognize hand written digits.

## Overview
This is a personal project which involved training a TensorFlow neural network to recognize hand written digits, deploying the model to the cloud, and creating a frontend user interface which allows people to draw their own digits on screen and recieve the model's prediction of what digit it thinks it sees. 

## Model Training
The tfTest folder contains all the code related to training and testing the model. Below are the steps I took to ensure the model worked properly before deploying it to the cloud.
### Training the Model
A TensorFlow sequential model was trained using 70,000 pixel arrays (28x28) which correspond to a 28x28 pixel image of a person's hand written digit. Each array element shows how light or dark a specific pixel is on the 28x28 image, depicted as a value somewhere between 0 and 1.

<strong><em>tfTest/trainModel.py</em></strong>
<br>
The training data is split into two randomized sets: a training set of 60,000 arrays and a testing set of 10,000 arrays.
```python
import tensorflow as tf

#Train Model
mnist = tf.keras.datasets.mnist

(X_train, Y_train),(X_test, Y_test) = mnist.load_data()
X_train = tf.keras.utils.normalize(X_train)
X_test = tf.keras.utils.normalize(X_test)
```

The model is initialized using the 'relu' activation function and form fitted to intake the 28x28 flattened pixel array. 
```python
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))
```

The model is compiled and fit to the training data and then evaluated with the test data.
```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=4)

loss, accuracy = model.evaluate(X_test, Y_test)
print(accuracy)
```
### Testing Real Images
I drew a digit and resized it to 28x28 using GNU's Image Manipulation Program (GIMP). Using the opencv-python package, the digit image file was read into memory and converted to an np array. The np array is passed to the model and a prediction is made and displayed using pyplot.

<strong><em>tfTest/loadModelTest.py</em></strong>
<br>
Draw an image on any drawing software and save it to the project directory. The digit must be black and must have a white background and resized to 28x28. 
<br>
![ksnip_20220221-125412](https://user-images.githubusercontent.com/74911365/155035435-1ffa50fb-8f0a-42cb-982c-f5c61667f2e5.png)

Add imports and load the model (loading the model allows you to use it without having to train it again).
```python
import tensorflow as tf
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
model = tf.keras.models.load_model("model")
```
Use opencv to read the image into memory and invert it to have black pixels show white, and white pixels show as black.
```python
image = cv.imread('testData/digit.png')[:, :, 0]
image = np.invert(np.array([image]))
```
The image will now look like this to the computer: <br>
![ksnip_20220221-130108](https://user-images.githubusercontent.com/74911365/155035882-9000bc24-7ef7-4809-8fac-776f8731fbb3.png)

Pass the inverted pixel array to the model and display the prediction.

```python
prediction = model.predict(image)
print('Interpretation: {}'.format(np.argmax(prediction)))
```
![ksnip_20220221-125833](https://user-images.githubusercontent.com/74911365/155035705-8a1161e7-c991-4b9d-b0a3-f6a6f12c27c9.png)


## Deploy to Google Cloud Platform (GCP) Function
The challenges associated with deploying a custom TensorFlow model to the cloud are discussed here. I started off with a trained and tested model. The end state is a RESTful Web API that can recieve HTTP POST requests containing image pixel data, and returns the model's prediction of what digit that pixel data represents.

### Saving the Weights in Cloud Storage
Cloud functions are a great way to bring RESTful functionality to any small script or program. Cloud functions often only have access to small amounts of volatile memory, which can't store a saved model indefinitly. Everytime the function's memory resets (which happens during periods of inactivity) the model needs to be recreated. To achieve this, the weights of the model needed to be saved and stored in non-volatile memory (Google Cloud Storage). This allowed the function to initialize itself by downloading the weights and creating the model after every time the function memory reset.

The weights saved in a Google Cloud Storage Bucket.
![image](https://user-images.githubusercontent.com/74911365/155038387-c4bcfcc0-446b-4a42-8d76-48c439bab87a.png)

### Function
This is the function's source code which lives in the cloud. The function's primary objective is to recieve pixel data via an HTTP request, and return the model's prediction.

<strong><em>tfTest/googleCloudFunction.py</em></strong>
<br>
Imports and a global model variable which is set to None (it will later be assigned to the actual TF model after it is initialized).

```python
import tensorflow as tf
from google.cloud import storage
import json
import numpy as np
import os
import tempfile

model = None
```

download_from_bucket function which can download blobs from an associated project bucket. This was very easy to do from a networking stand point because all GCP functions have an environment variable called 'GOOGLE_APPLICATION_CREDENTIALS', which contains all authentication information, and is automatically loaded by the google.cloud.storage client. It just works.

```python
def download_from_bucket(bucketName, blobName, fileDestination):
    try:
      storage_client = storage.Client()
      bucket = storage_client.get_bucket(bucketName)
      blob = bucket.blob(blobName)
      blob.download_to_filename(fileDestination)
      print("Blob {} downloaded to {}.".format(blobName, fileDestination))
    except:
      print("Download failed")
 ```


## Web Server

### Frontend
### Backend

## Deploy to Heroku

## Summary
