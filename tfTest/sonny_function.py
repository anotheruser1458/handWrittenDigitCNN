import tensorflow as tf
from google.cloud import storage
import json
import numpy as np

model = None

def download_from_bucket(bucketName, blobName, fileDestination):
    try:
      storage_client = storage.Client()
      bucket = storage_client.get_bucket(bucketName)
      blob = bucket.blob(blobName)
      blob.download_to_filename(fileDestination)
      print("Blob {} downloaded to {}.".format(blobName, fileDestination))
    except:
      print("Download failed")

def download_model():
    download_from_bucket("sonny-bucket", "weights/hwdWeights.index", "/tmp/weights/hwdWeights.index")
    download_from_bucket("sonny-bucket", "weights/hwdWeights.data-00000-of-00001", "/tmp/weights/hwdWeights.data-00000-of-00001")

def get_model():
    global model
    if model is not None:
      return
    else:
      download_model()
      model = tf.keras.models.Sequential()
      model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
      model.add(tf.keras.layers.Dense(units=128, activation='relu'))
      model.add(tf.keras.layers.Dense(units=128, activation='relu'))
      model.add(tf.keras.layers.Dense(units=128, activation='relu'))
      model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))
      model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
      model.load_weights('/tmp/weights/hwdWeights')

def sonny_hwd_guesser(request):
    response = {}
    # load json
    request_json = request.get_json()
    if not request_json:
      response['error'] = "no json content"
    get_model()
    image = request_json['image']
    image = np.asarray(image)
    print(image)
    print(type(image))
    prediction = np.argmax(model.predict(image))
    response['guess'] = str(prediction)
    return json.dumps(response)