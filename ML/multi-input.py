from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow import keras
# from tensorflow.keras.layers.convolutional import Conv2D
# from tensorflow.keras.layers.pooling import MaxPooling2D
# from tensorflow.keras.layers.merge import concatenate
from pathlib import Path
import numpy as np
from keras.utils import to_categorical

segmented_dict = [ 'sky', 'terrrain', 'tree', 'house', 'mycar', 'otherCar', 'lane4', 'lane3', 'lane2', 'lane1', ]

# Config
trainDirectory = 'Data/images_train/numpy'
pointsDataDirectory = 'Data/images_train/points/'
pointsTrainingDirectory = 'Data/points_train/'
pointsLabelDirectory = 'Data/points_label/'
numberOfLanes = 4
imgWidth = 224
imgHeight = 128
lrs = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]
EPOCHS = 10
BATCH_SIZE = 64

for filePath in Path(trainDirectory).glob('*.npy'):
  label = str(int(filePath.parts[-1][:-4]))
  imageAsArray = np.load(filePath, allow_pickle=True)
  # Normalization and Flattening/Reshaping
  imageAsArray = imageAsArray/255.0
  imageAsArray = imageAsArray.reshape(imgHeight, imgWidth, 3)

  points_data = np.load(pointsDataDirectory+label+'.npy', allow_pickle=True)
  # points_training_data = np.load(pointsTrainingDirectory+label+'.npy', allow_pickle=True) # points_label_data = np.load(pointsLabelDirectory+label+'.npy', allow_pickle=True) # for point_data,point_label in zip(points_training_data, points_label_data): # pixel_data = point_data[0]
  point_coordinate = np.asarray([point[0] for point in points_data])
  # point_coordinate = point_coordinate.reshape(-1, 2)
  point_label = np.asarray([point[1] for point in points_data])
  print(point_coordinate.shape)
  print(point_label.shape)
  # for point_coordinate,point_label in points_data: #   print(imageAsArray.shape) #   print(point_coordinate.shape) #   print(point_label)
    # first input model
  visible1 = Input(shape=(imgHeight, imgWidth, 3))
  conv11 = keras.layers.Conv2D(32, kernel_size=4, activation='relu')(visible1)
  pool11 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv11)
  conv12 = keras.layers.Conv2D(16, kernel_size=4, activation='relu')(pool11)
  pool12 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv12)
  flat1 = Flatten()(pool12)
  # second input model
  visible2 = Input(shape=(2))
  # conv21 = keras.layers.Conv2D(32, kernel_size=4, activation='relu')(visible2)
  # pool21 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv21)
  # conv22 = keras.layers.Conv2D(16, kernel_size=4, activation='relu')(pool21)
  # pool22 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv22)
  # flat2 = Flatten()(pool22)
  flat2 = Flatten()(visible2)
  # merge input models
  merge = keras.layers.concatenate([flat1, flat2])
  # merge = keras.layers.concatenate([flat1, visible2])
  # interpretation model
  hidden1 = Dense(10, activation='relu')(merge)
  hidden2 = Dense(10, activation='relu')(hidden1)
  output = Dense(10, activation='sigmoid')(hidden2)
  model = Model(inputs=[visible1, visible2], outputs=output)
  # summarize layers
  print(model.summary())
  sgd = keras.optimizers.SGD(lr=lrs[2])
  model.compile(optimizer=sgd,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  imagesAsArray = [imageAsArray for _ in point_coordinate]
  npLabels = np.array([segmented_dict.index(label) for label in point_label])
  npLabelsHotVector = to_categorical(npLabels)
  print(npLabelsHotVector)
  # npLabelsHotVector[np.arange(npLabels.size), "sky"] = 1
  model.fit([imagesAsArray,point_coordinate], npLabelsHotVector, epochs=EPOCHS, batch_size=BATCH_SIZE)
  break
  break
    

