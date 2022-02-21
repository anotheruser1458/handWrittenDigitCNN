import tensorflow as tf
import cv2 as cv
import numpy as np
import json

mnist = tf.keras.datasets.mnist

(X_train, Y_train),(X_test, Y_test) = mnist.load_data()
X_train = tf.keras.utils.normalize(X_train)
X_test = tf.keras.utils.normalize(X_test)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.load_weights('weights/hwdWeights')

loss, accuracy = model.evaluate(X_test, Y_test)
print(accuracy)

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
print(image.shape)
