# import numpy as np
# from tensorflow.keras.preprocessing.image import load_img
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.applications.vgg16 import preprocess_input
# from tensorflow.keras.applications.vgg16 import decode_predictions
# from tensorflow.keras.applications.vgg16 import VGG16
# # load the model
# model = VGG16()
# # load an image from file
# image = load_img('car.jpg', target_size=(224, 224))
# # convert the image pixels to a numpy array
# image = img_to_array(image)
# # reshape data for the model
# image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# # prepare the image for the VGG model
# image = preprocess_input(image)
# # predict the probability across all output classes
# yhat = model.predict(image)
# # convert the probabilities to class labels
# label = decode_predictions(yhat)

# for item in label:
#   print(item)

# # retrieve the most likely result, e.g. highest probability
# label = label[0][0]
# # print the classification
# print('%s (%.2f%%)' % (label[1], label[2]*100))


from keras import applications, layers, optimizers, Input
# Creating new model. Please note that this is NOT a Sequential() model.
from keras.models import Model
import numpy as np, numpy.random as npr

# Config
imagesAsArray = np.load("imagesAsArrayColor.npy")
labels = np.load("labelsVectorColor.npy")
numberOfTrainingImages = 12000
numberOfLanes = 4
imgWidth = 256
imgHeight = 144

lrs = [0.05]
batch_sizes = [32]
EPOCHS = 30

# Normalization and Flattening/Reshaping
imagesAsArray = imagesAsArray/255.0
imagesAsArray = imagesAsArray.reshape(-1, imgHeight, imgWidth,3)

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
imagesAsArray = None
# Logging config
print('Class Distribution: {}'.format(trainl.sum(axis=0)))
print('Training Data Dimension: {}'.format(traind.shape))
print('Training Label Dimension: {}'.format(trainl.shape))
print('Testing Data Dimension: {}'.format(testd.shape))
print('Testing Label Dimension: {}'.format(testl.shape))
print('Optimizer: SGD')
print('EPOCHS: {}'.format(EPOCHS))

for BATCH_SIZE in batch_sizes:
  for LR in lrs:

    print('BATCH_SIZE: {}'.format(BATCH_SIZE))
    print('Learning Rate: {}'.format(LR))

    input_tensor = Input(shape=(256, 144, 3))
    vgg_model = applications.VGG16(weights='imagenet',
                                  include_top=False,
                                  input_shape=(256, 144, 3))
    # Creating dictionary that maps layer names to the layers
    layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])
    print(layer_dict)
    # Getting output tensor of the last VGG layer that we want to include
    x = layer_dict['block2_pool'].output

    # Stacking a new simple convolutional network on top of it    
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4, activation='softmax')(x)

    model = Model(input=vgg_model.input, output=x)

    # Make sure that the pre-trained bottom layers are not trainable
    for layer in model.layers[:7]:
        layer.trainable = False

    sgd = optimizers.SGD(lr=LR)
    adam = optimizers.Adam(lr=LR)
    # Do not forget to compile it
    model.compile(loss=sgd,
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # To see the models' architecture and layer names, run the following
    vgg_model.summary()
    model.summary()

    # l=[np.where(r==1)[0][0] for r in labels]
    model.fit(traind, trainl, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # Evaluate the model on the test data using `evaluate`
    print('\n# Evaluate on test data')
    results = model.evaluate(testd, testl, batch_size=BATCH_SIZE)
    print('test loss, test acc:', results)

    # Generate predictions (probabilities -- the output of the last layer) on new data using `predict`
    print('\n# Generate predictions for 3 samples')
    predictions = model.predict(testd[:3])
    print('predictions:', predictions)
    print('actual', testl[:3])


# import numpy as np, numpy.random as npr
# import tensorflow as tf
# from tensorflow.keras import applications, layers, Input
# from tensorflow.keras.models import Model
# IMG_SHAPE = (256, 144, 3)
# # vgg = applications.VGG16(weights='imagenet', include_top=False, input_shape=IMG_SHAPE)
# vgg = applications.VGG16(weights='imagenet', include_top=False, input_shape=IMG_SHAPE)
# vgg.trainable = False
# # vgg.summary()
# # Creating dictionary that maps layer names to the layers
# layer_dict = dict([(layer.name, layer) for layer in vgg.layers])
# print(layer_dict)
# model = tf.keras.Sequential([
#   vgg,
#   # layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
#   # layers.MaxPooling2D(pool_size=(2, 2)),
#   layers.Flatten(),
#   layers.Dense(256, activation='relu'),
#   layers.Dense(256, activation='relu'),
#   # layers.Dropout(0.5),
#   layers.Dense(4, activation='softmax'),
# ])

# model.compile(optimizer=tf.keras.optimizers.Adam(),
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# model.summary()


# import tensorflow as tf, numpy as np, numpy.random as npr
# from tensorflow import keras
# from tensorflow.python.keras.callbacks import TensorBoard;
# from time import time;

# # Config
# imagesAsArray = np.load("imagesAsArrayColor.npy")
# labels = np.load("labelsVectorColor.npy")
# numberOfTrainingImages = 12000
# numberOfLanes = 4
# imgWidth = 256
# imgHeight = 144

# lrs = [0.05]
# batch_sizes = [32]
# EPOCHS = 30

# # Normalization and Flattening/Reshaping
# imagesAsArray = imagesAsArray/255.0
# imagesAsArray = imagesAsArray.reshape(-1, imgHeight, imgWidth,3)

# # Shuffling and Splitting training & testing data
# N = imagesAsArray.shape[0]
# rawIndices = np.array(range(0,N))
# npr.shuffle(rawIndices)
# trainIndice = rawIndices[0:numberOfTrainingImages]
# testIndice = rawIndices[numberOfTrainingImages:N]
# traind = imagesAsArray[trainIndice]
# trainl = labels[trainIndice]
# testd = imagesAsArray[testIndice]
# testl = labels[testIndice]
# imagesAsArray = None
# # Logging config
# print('Class Distribution: {}'.format(trainl.sum(axis=0)))
# print('Training Data Dimension: {}'.format(traind.shape))
# print('Training Label Dimension: {}'.format(trainl.shape))
# print('Testing Data Dimension: {}'.format(testd.shape))
# print('Testing Label Dimension: {}'.format(testl.shape))
# print('Optimizer: SGD')
# print('EPOCHS: {}'.format(EPOCHS))

# for BATCH_SIZE in batch_sizes:
#   for LR in lrs:

#     print('BATCH_SIZE: {}'.format(BATCH_SIZE))
#     print('Learning Rate: {}'.format(LR))

#     # vgg = applications.VGG16(weights='imagenet', include_top=False, input_shape=IMG_SHAPE)
#     vgg = keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(imgHeight, imgWidth,3))
#     vgg.trainable = False
#     # vgg.summary()
#     # Creating dictionary that maps layer names to the layers
#     # layer_dict = dict([(layer.name, layer) for layer in vgg.layers])
#     # print(layer_dict)
#     model = keras.Sequential([
#       vgg,
#       # keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
#       # keras.layers.MaxPooling2D(pool_size=(2, 2)),
#       keras.layers.Flatten(),
#       # keras.layers.Dense(256, activation='relu'),
#       # keras.layers.Dense(256, activation='relu'),
#       # keras.layers.Dropout(0.5),
#       keras.layers.Dense(4, activation='softmax'),
#     ])

#     tensorBoard = TensorBoard(log_dir="logs-vgg/{}SGD-LR_{}-BS_{}".format(time(), LR, BATCH_SIZE))

#     sgd = keras.optimizers.SGD(lr=LR)
#     adam = keras.optimizers.Adam()
#     model.compile(optimizer=sgd,
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])

#     # l=[np.where(r==1)[0][0] for r in labels]
#     model.fit(traind, trainl, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[tensorBoard])
#     model.summary()

#     # Evaluate the model on the test data using `evaluate`
#     print('\n# Evaluate on test data')
#     results = model.evaluate(testd, testl, batch_size=BATCH_SIZE)
#     print('test loss, test acc:', results)

#     # Generate predictions (probabilities -- the output of the last layer) on new data using `predict`
#     print('\n# Generate predictions for 3 samples')
#     predictions = model.predict(testd[:3])
#     print('predictions:', predictions)
#     print('actual', testl[:3])
