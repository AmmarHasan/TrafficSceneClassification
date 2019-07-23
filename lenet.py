import tensorflow as tf, numpy as np, numpy.random as npr
from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard;
from time import time;
from PIL import Image

# Config
imagesAsArray = np.load("imagesAsArray.npy")
labels = np.load("labelsAll.npy")
numberOfTrainingImages = 12000
numberOfLanes = 4
imgWidth = 256
imgHeight = 144
neuronsInImage = imgWidth*imgHeight

# # Display Image
# print(imagesAsArray[0])
# img = Image.fromarray(imagesAsArray[0])
# img.show()
# print(trainl[0])
# print(imagesAsArray[10])
# img = Image.fromarray(imagesAsArray[10])
# img.show()
# print(trainl[10])

# Normalization and Flattening/Reshaping
imagesAsArray = imagesAsArray/255.0
imagesAsArray = imagesAsArray.reshape(-1, 144, 256,1)

# Shuffling and Splitting training & testing data
N = imagesAsArray.shape[0]
rawIndices = np.array(range(0,N))
npr.shuffle(rawIndices)
trainIndice = rawIndices[0:numberOfTrainingImages]
testIndice = rawIndices[numberOfTrainingImages:N]
traind = imagesAsArray[trainIndice]
trainl = labels[trainIndice]
testd = imagesAsArray[testIndice]
testl = labels[testIndice]

# Logging config
unique, counts = np.unique(trainl, return_counts=True)
print('Class Distribution: {}'.format(dict(zip(unique, counts))))
print('Training Data Dimension: {}'.format(traind.shape))
print('Training Label Dimension: {}'.format(trainl.shape))
print('Testing Data Dimension: {}'.format(testd.shape))
print('Testing Label Dimension: {}'.format(testl.shape))

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
tensorBoard = TensorBoard(log_dir="logs/{}".format(time()))

sgd = keras.optimizers.SGD(lr=0.01)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# l=[np.where(r==1)[0][0] for r in labels]
model.fit(traind, trainl, epochs=20, batch_size=60, callbacks=[tensorBoard])

# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
results = model.evaluate(testd, testl, batch_size=128)
print('test loss, test acc:', results)

# Generate predictions (probabilities -- the output of the last layer) on new data using `predict`
print('\n# Generate predictions for 3 samples')
predictions = model.predict(testd[:3])
print('predictions:', predictions)
