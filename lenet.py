import tensorflow as tf;
from tensorflow import keras;
import numpy as np;
import numpy.random as npr;
import _pickle as cPickle
from PIL import Image;
import os

imagesAsArray = np.load("imagesAsArray.npy")
labels = np.load("labelsAll.npy")
print(imagesAsArray.shape)
print(labels.shape)

N = imagesAsArray.shape[0];
rawIndices = np.array(range(0,N));
npr.shuffle(rawIndices);
trainIndice = rawIndices[0:4000];
testIndice = rawIndices[4000:6000];
traind = imagesAsArray[trainIndice];
trainl = labels[trainIndice];
testd = imagesAsArray[testIndice];
testl = labels[testIndice];

# print(traind[0])
# img = Image.fromarray( traind[0])
# img.show()
# print(trainl[0])
# print(traind[10])
# img = Image.fromarray( traind[10])
# img.show()
# print(trainl[10])

numberOfLanes = 4
imgWidth = 256;
imgHeight = 144;
neuronsInImage = imgWidth*imgHeight;
traind = traind.reshape(-1, 144, 256,1);
testd = testd.reshape(-1, 144, 256,1);
print(traind.shape)
print(trainl.shape)
unique, counts = np.unique(trainl, return_counts=True)
print(dict(zip(unique, counts)))

model = keras.models.Sequential([
  keras.layers.Conv2D(name='FirstConv2D', filters=6, kernel_size=(3, 3), activation='relu', padding="SAME"),
  keras.layers.AveragePooling2D(name='FirstAvgPool', padding="SAME"),
  keras.layers.Conv2D(name='SecondConv2D', filters=16, kernel_size=(3, 3), activation='relu', padding="SAME"),
  keras.layers.AveragePooling2D(name='SecAvgPool', padding="SAME"),
  keras.layers.Flatten(),
  keras.layers.Dense(name='FirstDense', units=120, activation='relu'),
  keras.layers.Dense(name='SecDense', units=84, activation='relu'),
  keras.layers.Dense(name='ThirdDense', units=numberOfLanes, activation = 'softmax')
])

sgd = keras.optimizers.SGD(lr=0.01)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# l=[np.where(r==1)[0][0] for r in labels]
# model.fit(imagesAsArray[0:8000], labels[0:8000], epochs=5)
model.fit(traind, trainl, epochs=20)
              
# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
results = model.evaluate(testd, testl, batch_size=128)
print('test loss, test acc:', results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print('\n# Generate predictions for 3 samples')
predictions = model.predict(testd[:3])
print('predictions:', predictions)
