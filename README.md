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

download_from_bucket function which can download blobs from an associated project bucket. This was very easy to do from a networking stand point because all GCP functions have an environment variable called 'GOOGLE_APPLICATION_CREDENTIALS', which contains all authentication information and is automatically loaded by the google.cloud.storage client. It just works.
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

Main function that is called when an API request is made. Variables initialized and the model is checked. If there is no model, then download the weighs into the tmp folder and initialize it.
```python
def sonny_hwd(request):
    global model
    response = {}
    if model == None:
      download_from_bucket("sonny-bucket", "weights/hwdWeights.data-00000-of-00001", "/tmp/hwdWeights.data-00000-of-00001")
      download_from_bucket("sonny-bucket", "weights/hwdWeights.index", "/tmp/hwdWeights.index")
      model = tf.keras.models.Sequential()
      model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
      model.add(tf.keras.layers.Dense(units=128, activation='relu'))
      model.add(tf.keras.layers.Dense(units=128, activation='relu'))
      model.add(tf.keras.layers.Dense(units=128, activation='relu'))
      model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))
      model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
      model.load_weights('/tmp/hwdWeights')
```

Once there is a working model, retrieve the request payload and covert it to json. Read the string data which is delimited by commas and convert every string to an integer.
```python
    request_json = request.get_json()
    image = request_json['image'].split(",")
    image = [int(x) for x in image]
```

Convert the python list into an np array and reshape to be 28x28. Pass the np array to the model and return the prediction in a json response. The cloud function will automatically add the return value from the function to the payload of the HTTP response.
```python
    image = np.array(image).reshape((1, 28, 28))
    prediction = np.argmax(model.predict(image))
    response['prediction'] = str(prediction)
    return json.dumps(response)
```
<!-- The web application's objective is to create an interface for users to draw a digit with their mouse and then display the digit that the model thinks it sees.  -->

### Web Application Frontend
The frontend's objective is to inform users of the functionality of the app and the neural network's capabilities. It also provides an interface (an html canvas) which allows users to draw a number with their mouse, and submit the drawing to the model. Below is an explanation of how the canvas interface was created.

<strong><em>templates/home.html</em></strong>
HTML canvas element added to homepage.
```html
<canvas id="canvas" style="background-color:white;" class="responsive"></canvas>
```
<img src="https://user-images.githubusercontent.com/74911365/155042928-d932fff0-efe6-4e99-bc2d-9cc2b97378d6.png" width=400>

Using javascript select the canvas, and associate it's context with it's height and width (the height and width are hard coded in CSS).
```javascript
canvas = document.querySelector("#canvas")
var ctx = canvas.getContext('2d');
ctx.canvas.width = 500;
ctx.canvas.height = 500;
```

Establish a position variable which will be updated whenever the mousemove, mousedown, and mouseenter events occur.
```javascript
var pos = { x: 0, y: 0 };

document.addEventListener('mousemove', draw);
document.addEventListener('mousedown', setPosition);
document.addEventListener('mouseenter', setPosition);
```

setPosition is called whenever the mouse button is clicked and updates the pos variable declared earlier. Moving the mouse calls the draw function which causes the canvas to draw on itself at the mouse's current X and Y coordinate. The drawing will only occur if the mouse is located somewhere on the canvas.

```javascript
function setPosition(e) {
    rect = canvas.getBoundingClientRect();
    x = rect.x;
    y = rect.y;
    pos.x = e.clientX - x;
    pos.y = e.clientY - y;
}

function draw(e) {
  // mouse left button must be pressed
  if (e.buttons !== 1) return;

  ctx.beginPath(); // begin

  ctx.lineWidth = 10;
  ctx.lineCap = 'round';
  ctx.strokeStyle = 'black';

  ctx.moveTo(pos.x, pos.y); // from
  setPosition(e);
  ctx.lineTo(pos.x, pos.y); // to

  ctx.stroke(); // draw it!
}
```
![image](https://user-images.githubusercontent.com/74911365/155043225-c01fce24-dcea-4d92-ae5f-5c1fda7e61d9.png)



### Web Application Backend
The backend's objective is to serve the static files to users, capture and clean the canvas data, send the cleaned data to the cloud function, and return the model's prediction back to the frontend.

## Deploy to Heroku
Heroku is the web host used for this application. A custom domain was purchased on namecheap.com which is now directly linked to the project. 

## Summary
