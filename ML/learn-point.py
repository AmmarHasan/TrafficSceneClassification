# # Add pixel coordinates to  fully connected layer tensorflow low-level API
import numpy as np
import numpy.random as npr
from keras.utils import to_categorical
from pathlib import Path
import tensorflow as tf

# Config
trainDirectory = 'Data/images_train/numpy'
pointsDataDirectory = 'Data/images_train/points/'
pointsTrainingDirectory = 'Data/points_train/'
pointsLabelDirectory = 'Data/points_label/'
numberOfTrainingImages = 6500
numberOfTestingImages = 1500
numberOfLanes = 4
totalClasses = 10
imgWidth = 224
imgHeight = 128
neuronsInImage = imgWidth*imgHeight*3
lrs = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]
EPOCHS = 60
BATCH_SIZE = 512
segmented_dict = [ 'sky', 'terrrain', 'tree', 'house', 'mycar', 'otherCar', 'lane4', 'lane3', 'lane2', 'lane1', ]

imagesAsArray = []
point_coordinates = []
labelsHotVectors = []
for filePath in Path(trainDirectory).glob('*.npy'):
  label = str(int(filePath.parts[-1][:-4]))
  imageAsArray = np.load(filePath, allow_pickle=True)
  # Normalization and Flattening/Reshaping
  imageAsArray = imageAsArray/255.0
  imageAsArray = imageAsArray.reshape(imgHeight, imgWidth, 3)

  points_data = np.load(pointsDataDirectory+label+'.npy', allow_pickle=True)
  point_coordinate = np.asarray([point[0] for point in points_data])
  point_label = np.asarray([point[1] for point in points_data])
  npLabels = np.array([segmented_dict.index(label) for label in point_label])
  npLabelsHotVector = to_categorical(npLabels)
  imagesAsArray.append(imageAsArray)
  point_coordinates.append(point_coordinate)
  labelsHotVectors.append(npLabelsHotVector)

imagesAsArrayDuplicate = []
for i in range(0, len(imagesAsArray)):
  imagesAsArrayDuplicate.append([imageAsArray for _ in point_coordinates[i]])

point_coordinates = np.array(point_coordinates).reshape(-1,2)
labelsHotVectors = np.array(labelsHotVectors).reshape(-1,10)
imagesAsArrayDuplicate = np.array(imagesAsArrayDuplicate).reshape(-1,neuronsInImage)
imagesAndCoordinate = np.concatenate((imagesAsArrayDuplicate,point_coordinates), axis=1)
print(imagesAndCoordinate.shape)


# ## Shuffling and Splitting training & testing data
imagesAsArrayDuplicate=None
imagesAsArray=None
rawIndices = np.array(range(0,labelsHotVectors.shape[0]))
npr.shuffle(rawIndices)
trainIndice = rawIndices[0:numberOfTrainingImages]
testIndice = rawIndices[numberOfTrainingImages: numberOfTrainingImages + numberOfTestingImages]
traind = imagesAndCoordinate[trainIndice]
trainl = labelsHotVectors[trainIndice]
testd = imagesAndCoordinate[testIndice]
testl = labelsHotVectors[testIndice]
point_coordinates=None
labelsHotVectors=None
imagesAndCoordinate=None
rawIndices=None

# ## Initialize data tensor in NHWC format
data_placeholder = tf.placeholder(tf.float32,[None,neuronsInImage+2])
label_placeholder = tf.placeholder(tf.float32,[None,totalClasses])

# compute activations
W = tf.Variable(tf.zeros(shape=(neuronsInImage+2,totalClasses)),dtype=tf.float32)
b = tf.Variable(tf.ones(shape=(1,totalClasses)),dtype=tf.float32)
# cp = tf.constant([coordinate_placeholder[0], coordinate_placeholder[1],0,0])
logits = tf.matmul(data_placeholder,W) + b #+ coordinate_placeholder;

# # ## Hidden Layer 1
# # #### Convolution Layer with 32 fiters and a kernel size of 5
# conv1 = tf.nn.relu(tf.layers.conv2d(data_placeholder,6, 5,name="H1"))
# print(conv1)

# # #### Max Pooling (down-sampling) with strides of 2 and kernel size of 2
# a1 = tf.layers.max_pooling2d(conv1, 2, 2)
# print(a1)

# # ## Hidden Layer 2
# # #### Convolution Layer with 64 filters and a kernel size of 3
# conv2 = tf.nn.relu(tf.layers.conv2d(a1, 16, 5,name="H2"))

# # #### Max Pooling (down-sampling) with strides of 2 and kernel size of 2
# a2 = tf.layers.max_pooling2d(conv2, 2, 2)
# print(a2)
# # a2flat = tf.reshape(a2, (-1,4*4*16))
# a2flat = tf.reshape(a2, (-1,33*61*16))
# print(a2flat)

# # ## Hidden Layer 3
# Z3 = 120
# # allocate variables
# # W3 = tf.Variable(npr.uniform(-0.01,0.01, [4*4*16,Z3]),dtype=tf.float32, name ="W3")
# W3 = tf.Variable(npr.uniform(-0.01,0.01, [33*61*16,Z3]),dtype=tf.float32, name ="W3")
# b3 = tf.Variable(npr.uniform(-0.01,0.01, [1,Z3]),dtype=tf.float32, name ="b3")
# # compute activations
# a3 = tf.nn.relu(tf.matmul(a2flat, W3) + b3)
# print(a3)

# # ## Hidden Layer 4
# Z4 = 84
# # allocate variables
# W4 = tf.Variable(npr.uniform(-0.01,0.01, [Z3,Z4]),dtype=tf.float32, name ="W4")
# b4 = tf.Variable(npr.uniform(-0.01,0.01, [1,Z4]),dtype=tf.float32, name ="b4")
# # compute activations
# a4 = tf.nn.relu(tf.matmul(a3, W4) + b4)
# print(a4)

# # ## Output layer
# # alloc variables
# Z5 = numberOfLanes
# W5 = tf.Variable(npr.uniform(-0.1,0.1, [Z4,Z5]),dtype=tf.float32, name ="W5")
# b5 = tf.Variable(npr.uniform(-0.01,0.01, [1,Z5]),dtype=tf.float32, name ="b5")
# # compute activations
# logits = tf.matmul(a4, W5) + b5
# print(logits)

# ## Loss and Accuracy functions
lossBySample = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=label_placeholder)
loss = tf.reduce_mean(lossBySample)

nrCorrect = tf.reduce_mean(tf.cast(tf.equal (tf.argmax(logits,axis=1), tf.argmax(label_placeholder,axis=1)), tf.float32))

# ## Create update optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate = lrs[2])
update = optimizer.minimize(loss)

# ## Learn
# Model Evaluation
def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(nrCorrect, feed_dict={data_placeholder: batch_x, label_placeholder: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

with tf.Session() as sess:
    ## init all variables
    sess.run(tf.global_variables_initializer())
    for i in range(EPOCHS):
        for offset in range(0, trainl.shape[0], BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = traind[offset:end], trainl[offset:end]
            batch_test_x, batch_test_y = testd[offset:end], testl[offset:end]
            trainFd = {data_placeholder: batch_x, label_placeholder: batch_y}
            testFd = {data_placeholder: batch_test_x, label_placeholder: batch_test_y}
            #update parameters
            sess.run(update, feed_dict=trainFd)
            correct, lossVal = sess.run([nrCorrect,loss], feed_dict=trainFd)
            #testacc = sess.run(nrCorrect, feed_dict = testFd)
            print('Epoch {}, acc={:.6f}, loss={:.6f}\r'.format(i, float(correct), lossVal), end='')

        print()
        X_validation, y_validation = traind[::-1], trainl[::-1]
        validation_accuracy = evaluate(X_validation, y_validation)
        print("Validation Accuracy = {:.6f}".format(validation_accuracy))
        test_accuracy = evaluate(testd, testl)
        print("Test Accuracy = {:.6f}".format(test_accuracy))
        print()
    
    test_accuracy = evaluate(testd, testl)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

