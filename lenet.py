import tensorflow as tf;
import numpy as np;
import _pickle as cPickle

imagesAsArrayFilename = 'imagesAsArray.data'
with open(imagesAsArrayFilename, 'rb') as filehandle:
    # read the data as binary data stream
    imagesAsArray = np.array(cPickle.load(filehandle))

labelsFileName = 'labels.data'
with open(labelsFileName, 'rb') as filehandle:
    # read the data as binary data stream
    labels = np.array(cPickle.load(filehandle))

numberOfLanes = 4
imgWidth = 256;
imgHeight = 144;
neuronsInImage = imgWidth*imgHeight;
imagesAsArray = imagesAsArray.reshape(-1, 144, 256,1);
print(imagesAsArray.shape)

lenet = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(name='FirstConv2D', filters=6, kernel_size=(3, 3), activation='relu'),
  tf.keras.layers.AveragePooling2D(name='FirstAvgPool'),
  tf.keras.layers.Conv2D(name='SecondConv2D', filters=16, kernel_size=(3, 3), activation='relu'),
  tf.keras.layers.AveragePooling2D(name='SecAvgPool'),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(name='FirstDense', units=120, activation='relu'),
  tf.keras.layers.Dense(name='SecDense', units=84, activation='relu'),
  tf.keras.layers.Dense(name='ThirdDense', units=numberOfLanes, activation = 'softmax')
])

sgd = tf.keras.optimizers.SGD(lr=0.2)
lenet.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# l=[np.where(r==1)[0][0] for r in labels]
lenet.fit(imagesAsArray[0:2000], labels[0:2000], epochs=5)






# imagesAsArray = imagesAsArray.reshape(-1, neuronsInImage);

# inputShape = (neuronsInImage,1) #ValueError: Input 0 of layer conv2d is incompatible with the layer: expected ndim=4, found ndim=3. Full shape received: [None, 36864, 1]
#inputShape = (1,neuronsInImage,1) #ValueError: Negative dimension size caused by subtracting 3 from 1 for 'conv2d/Conv2D' (op: 'Conv2D') with input shapes: [?,1,36864,1], [3,3,1,6].

# inputShape = (imgHeight,imgWidth,1) #ValueError: Error when checking input: expected FirstConv2D_input to have 4 dimensions, but got array with shape (2000, 144, 256)
# inputShape = (None,imgHeight,imgWidth,1) #ValueError: Input 0 of layer FirstConv2D is incompatible with the layer: expected ndim=4, found ndim=5. Full shape received: [None, None, 144, 256, 1]
# inputShape = (imgHeight,imgWidth) #ValueError: Input 0 of layer FirstConv2D is incompatible with the layer: expected ndim=4, found ndim=3. Full shape received: [None, 144, 256]

# inputShape = (32,32,1) 
# imagesAsArray = imagesAsArray.reshape(-1, 32,32,1);

# mnist = tf.keras.datasets.mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(144, 256)),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(10, activation='softmax')
# ])

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=5)

# test_loss, test_acc = model.evaluate(x_test, y_test)

# print('\nTest accuracy:', test_acc)
# print('\nTest loss:', test_loss)
