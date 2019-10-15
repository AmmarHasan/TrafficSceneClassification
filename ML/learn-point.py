# # Add pixel coordinates to  fully connected layer tensorflow low-level API
import numpy as np
import numpy.random as npr
from keras.utils import to_categorical
from pathlib import Path
import tensorflow as tf
from PIL import Image

# Config
imgWidth,imgHeight,imageType = (256, 144, 'png')
full_res_image_path = 'Data/ScreenCapture-3'
annotationDirectory = full_res_image_path+'/images_segmented-'+str(imgWidth)+'x'+str(imgHeight)+'-'+imageType  # -224x128-png/'
trainDirectory = full_res_image_path+'/images_train-'+str(imgWidth)+'x'+str(imgHeight)+'-'+imageType
pointsDataDirectory = trainDirectory + '/points-1000/'
numberOfTrainingImages = 6500
numberOfTestingImages = 1500
segmented_dict = [ 'outside', 'lane1', 'lane2', 'lane3', 'lane4', ]
totalClasses = len(segmented_dict)
neuronsInImage = imgWidth*imgHeight*3
lrs = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]
EPOCHS = 1000
BATCH_SIZE = 128

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def getModel():
    # ## Initialize data tensor in NHWC format
  data_placeholder = tf.placeholder(tf.float32,[None,imgHeight, imgWidth,3])
  label_placeholder = tf.placeholder(tf.float32,[None,totalClasses])

  # ## Hidden Layer 1
  # #### Convolution Layer with 32 fiters and a kernel size of 5
  conv1 = tf.nn.relu(tf.layers.conv2d(data_placeholder,6, 5,name="H1"))
  # print(conv1)

  # #### Max Pooling (down-sampling) with strides of 2 and kernel size of 2
  a1 = tf.layers.max_pooling2d(conv1, 2, 2)
  # print(a1)

  # ## Hidden Layer 2
  # #### Convolution Layer with 64 filters and a kernel size of 3
  conv2 = tf.nn.relu(tf.layers.conv2d(a1, 16, 5,name="H2"))
  # print(conv2)

  # #### Max Pooling (down-sampling) with strides of 2 and kernel size of 2
  a2 = tf.layers.max_pooling2d(conv2, 2, 2)
  # print(a2)
  # For 28x28:
  # a2flat = tf.reshape(a2, (-1,4*4*16))
  # For 144x256:
  a2flat = tf.reshape(a2, (-1,33*61*16)) 
  # For 128x224:
  # a2flat = tf.reshape(a2, (-1,29*53*16)) 
  # print(a2flat)

  # ## Hidden Layer 3
  Z3 = 120
  # allocate variables
  # W3 = tf.Variable(npr.uniform(-0.01,0.01, [4*4*16,Z3]),dtype=tf.float32, name ="W3")
  W3 = tf.Variable(npr.uniform(-0.01,0.01, [33*61*16,Z3]),dtype=tf.float32, name ="W3")
  b3 = tf.Variable(npr.uniform(-0.01,0.01, [1,Z3]),dtype=tf.float32, name ="b3")
  # compute activations
  a3 = tf.nn.relu(tf.matmul(a2flat, W3) + b3)
  # print(a3)

  # ## Hidden Layer 4
  Z4 = 84
  # allocate variables
  W4 = tf.Variable(npr.uniform(-0.01,0.01, [Z3,Z4]),dtype=tf.float32, name ="W4")
  b4 = tf.Variable(npr.uniform(-0.01,0.01, [1,Z4]),dtype=tf.float32, name ="b4")
  # compute activations
  a4 = tf.nn.relu(tf.matmul(a3, W4) + b4)
  # print(a4)

  # ## Output layer
  # alloc variables
  Z5 = totalClasses
  W5 = tf.Variable(npr.uniform(-0.1,0.1, [Z4,Z5]),dtype=tf.float32, name ="W5")
  b5 = tf.Variable(npr.uniform(-0.01,0.01, [1,Z5]),dtype=tf.float32, name ="b5")
  # compute activations
  logits = tf.matmul(a4, W5) + b5
  # print(logits)

  # ## Loss and Accuracy functions
  lossBySample = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=label_placeholder)
  loss = tf.reduce_mean(lossBySample)

  nrCorrect = tf.reduce_mean(tf.cast(tf.equal (tf.argmax(logits,axis=1), tf.argmax(label_placeholder,axis=1)), tf.float32))

  # ## Create update optimizer
  optimizer = tf.train.GradientDescentOptimizer(learning_rate = lrs[2])
  update = optimizer.minimize(loss)
  return (update, nrCorrect, loss, data_placeholder, label_placeholder)
  
# ## Learn
# Model Evaluation
def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    total_loss = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy, lossVal = sess.run([nrCorrect,loss], feed_dict={data_placeholder: batch_x, label_placeholder: batch_y})
        total_accuracy += (accuracy * len(batch_x))
        total_loss += (lossVal * len(batch_x))
    return (total_accuracy / num_examples, total_loss / num_examples)

batch_x = []
batch_y = []
batch_counter = 0
img_paths = []
for filePath in Path(trainDirectory).glob('*.png'):
  img_paths.append(filePath)

shuffled_img_paths=[]
for img_path in img_paths:
  points = np.load(pointsDataDirectory+img_path.stem+'.npy',allow_pickle=True)
  for i in range(0,len(points)):
    point_coordinate, point_label = points[i]
    npLabels = segmented_dict.index(point_label)
    npLabelsHotVector = to_categorical(npLabels, num_classes=len(segmented_dict))
    shuffled_img_paths.append((i, point_coordinate,  npLabelsHotVector, img_path))

batch_img_paths = []
npr.shuffle(shuffled_img_paths)
for x in batch(shuffled_img_paths, BATCH_SIZE):
    batch_img_paths.append(x)

# len(batch_img_paths) -> 100153
twenty_percent = int(100153*0.2)
print(twenty_percent)
# test_img_points_paths = batch_img_paths[:twenty_percent]
# train_img_points_paths = batch_img_paths[twenty_percent:]
test_img_points_paths = batch_img_paths[:8]
train_img_points_paths = batch_img_paths[8:10]

img_paths = None
shuffled_img_paths = None
batch_img_paths = None

update, nrCorrect, loss, data_placeholder, label_placeholder = getModel()
with tf.Session() as sess:
    ## init all variables
    sess.run(tf.global_variables_initializer())
    for epoch in range(EPOCHS):
      for i in range(len(train_img_points_paths)-1):
          batch_x = []
          batch_y = []
          for j in range(BATCH_SIZE):
              _, point_coordinate, npLabelsHotVector, img_path = train_img_points_paths[i][j]
              img = np.array(Image.open(img_path))
              img = img / 255.0                                             # Normalization and Flattening/Reshaping
              img[imgHeight-1, imgWidth-1] = list(point_coordinate)         # Put pixel coordinates in last pixel of image
              batch_x.append(img)
              batch_y.append(npLabelsHotVector)

          trainFd = {data_placeholder: batch_x, label_placeholder: batch_y}
          sess.run(update, feed_dict=trainFd)                               # Update parameters
          correct, lossVal = sess.run([nrCorrect,loss], feed_dict=trainFd)
          #testacc = sess.run(nrCorrect, feed_dict = testFd)
          print('Epoch {}, acc={:.6f}, loss={:.6f}\r'.format(epoch, float(correct), lossVal), end='')
      print()
