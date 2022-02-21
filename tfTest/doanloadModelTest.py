# DONT FORGET TO PUT REQUIREMENTS IN CLOUD

from google.cloud import storage
import os
import tensorflow as tf
import numpy as np
import cv2 as cv

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "../sonny-service-account-key.json"

storage_client = storage.Client()
bucket = storage_client.get_bucket('sonny-bucket')

blob = bucket.blob('weights/hwdWeights.index')
blob2 = bucket.blob('weights/hwdWeights.data-00000-of-00001')
blob.download_to_filename('downloadWeights/hwdWeights.index')
blob2.download_to_filename("downloadWeights/hwdWeights.data-00000-of-00001")

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.load_weights('downloadWeights/hwdWeights')

image = cv.imread('digit.png')[:, :, 0]
image = np.invert(np.array([image]))

numData = image[0]
newNumData = []

for x in numData:
    for y in x:
        newNumData.append(str(y))

# js = json.dumps({'data':newNumData})
simJs = {"data": newNumData}
image = simJs['data']
image = [int(x) for x in image]
narr = np.array(image)
image = narr.reshape((1, 28, 28))

prediction = model.predict(image)
print('Interpretation: {}'.format(np.argmax(prediction)))
