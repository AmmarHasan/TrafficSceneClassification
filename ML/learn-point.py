# # Add pixel coordinates to  fully connected layer tensorflow low-level API
import numpy as np
import numpy.random as npr
from keras.utils import to_categorical
from pathlib import Path
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# import _pickle as cPickle


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
EPOCHS = 20
BATCH_SIZE = 64
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

# if (numberOfTrainingImages + numberOfTestingImages >= imagesAsArray.shape[0]):
#   print('Changed number of trainig and testing data')
#   numberOfTrainingImages = 1000
#   numberOfTestingImages = 3265

# ## Shuffling and Splitting training & testing data
# In[ ]:
rawIndices = np.array(range(0,labelsHotVectors.shape[0]))
npr.shuffle(rawIndices)
trainIndice = rawIndices[0:numberOfTrainingImages]
testIndice = rawIndices[numberOfTrainingImages: numberOfTrainingImages + numberOfTestingImages]
traind = imagesAndCoordinate[trainIndice]
trainl = labelsHotVectors[trainIndice]
testd = imagesAndCoordinate[testIndice]
testl = labelsHotVectors[testIndice]
imagesAsArray=None
point_coordinates=None
labelsHotVectors=None
imagesAsArrayDuplicate=None
imagesAndCoordinate=None
rawIndices=None

# # ## Logging configuration
# # In[ ]:
# print('Class Distribution: {}'.format(trainl.sum(axis=0)))
# print('Training Data Dimension: {}'.format(traind.shape))
# print('Training Label Dimension: {}'.format(trainl.shape))
# print('Testing Data Dimension: {}'.format(testd.shape))
# print('Testing Label Dimension: {}'.format(testl.shape))
# print('EPOCHS: {}'.format(EPOCHS))
# print('BATCH_SIZE: {}'.format(BATCH_SIZE))
# print('Optimizer: SGD')
# print('Learning Rate: {}'.format(LR))

# # ## Flatten/Reshape Input
# # In[ ]:
# traind = traind.reshape(-1, imgHeight*imgWidth*1)
# testd = testd.reshape(-1, imgHeight*imgWidth* 1)
# print(traind.shape, testd.shape)


# ## Initialize data tensor in NHWC format
data_placeholder = tf.placeholder(tf.float32,[None,neuronsInImage+2])
label_placeholder = tf.placeholder(tf.float32,[None,totalClasses])
# coordinate_placeholder = tf.placeholder(tf.float32,[None,totalClasses])
# black_pixel=tf.Variable(tf.zeros(shape=(3)),dtype=tf.float32)
# sess.run(tf.concat([dp,[b]],0),feed_dict={dp:[[1,2,3], [4,5,6]]})
# sess.run(var,feed_dict={var:[1,2,3]})
# tf.concat([[b], data_placeholder],0),feed_dict={dp:[[1,2,3], [4,5,6]]})

# compute activations
W = tf.Variable(tf.zeros(shape=(neuronsInImage+2,totalClasses)),dtype=tf.float32)
b = tf.Variable(tf.ones(shape=(1,totalClasses)),dtype=tf.float32)
# cp = tf.constant([coordinate_placeholder[0], coordinate_placeholder[1],0,0])
logits = tf.matmul(data_placeholder,W) + b #+ coordinate_placeholder;

# ## Loss and Accuracy functions
# In[ ]:
lossBySample = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=label_placeholder)
loss = tf.reduce_mean(lossBySample)

nrCorrect = tf.reduce_mean(tf.cast(tf.equal (tf.argmax(logits,axis=1), tf.argmax(label_placeholder,axis=1)), tf.float32))

# ## Create update optimizer
# In[ ]:
optimizer = tf.train.GradientDescentOptimizer(learning_rate = lrs[2])
update = optimizer.minimize(loss)

# ## Learn
# In[ ]:
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

# In[ ]:
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
    
#     test_accuracy = evaluate(testd, testl)
#     print("Test Accuracy = {:.3f}".format(test_accuracy))

