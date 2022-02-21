#change digit.png with hand drawn digit, 28x28 pixels
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv
import PIL

#Train Model
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
model.fit(X_train, Y_train, epochs=4)

loss, accuracy = model.evaluate(X_test, Y_test)
print(accuracy)

#Load model test and prediction test
model = tf.keras.models.load_model("model")
model.save_weights("hwdWeights")
image = cv.imread('digit.png')[:, :, 0]
image = np.invert(np.array([image]))
prediction = model.predict(image)
print('Interpretation: {}'.format(np.argmax(prediction)))
plt.imshow(image[0])
plt.show()

