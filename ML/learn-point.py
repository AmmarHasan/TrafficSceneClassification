# # Add pixel coordinates to  fully connected layer tensorflow low-level API
import numpy as np
import numpy.random as npr
from keras.utils import to_categorical
from pathlib import Path
import tensorflow as tf
from PIL import Image
import time
import pickle
import os
import tensorboard
import datetime

def createDirectory(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)
  return directory

# Config
imgWidth,imgHeight,imageType = (256, 144, 'png')
full_res_image_path = 'Data/ScreenCapture-3'
trainDirectory = full_res_image_path+'/images_train-'+str(imgWidth)+'x'+str(imgHeight)+'-'+imageType
data_filename='-test'
data_path = f'{full_res_image_path}/shuffled_img_paths{data_filename}.p'
save_directory=createDirectory(f'{full_res_image_path}/tmp{data_filename}-adam')
result=f'{save_directory}/result.txt'
log_dir=createDirectory(f'{save_directory}/log')
lrs = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]
LR = lrs[4]
classes = [ 'outside', 'lane1', 'lane2', 'lane3', 'lane4']
totalClasses = len(classes)
EPOCHS = 15
BATCH_SIZE = 128

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def getModel():
    # ## Initialize data tensor in NHWC format
  with tf.variable_scope('dataset_inputs') as scope:
    data_placeholder = tf.placeholder(tf.float32,[None,imgHeight, imgWidth,3], name='Images')
    label_placeholder = tf.placeholder(tf.float32,[None,totalClasses], name='Labels')

  # n_t = tf.constant(n, dtype=tf.float32)
  # data_placeholder = tf.math.multiply(data_placeholder, n_t)
  # ## Hidden Layer 1
  # #### Convolution Layer with 32 fiters and a kernel size of 5
  conv1 = tf.nn.relu(tf.layers.conv2d(data_placeholder,6, 5,name="Conv_H1"))
  # print(conv1)

  # #### Max Pooling (down-sampling) with strides of 2 and kernel size of 2
  a1 = tf.layers.max_pooling2d(conv1, 2, 2, name="Maxpooling_1")
  # print(a1)

  # ## Hidden Layer 2
  # #### Convolution Layer with 64 filters and a kernel size of 3
  conv2 = tf.nn.relu(tf.layers.conv2d(a1, 16, 5,name="Conv_H2"))
  # print(conv2)

  # #### Max Pooling (down-sampling) with strides of 2 and kernel size of 2
  a2 = tf.layers.max_pooling2d(conv2, 2, 2, name="Maxpooling_2")

  # ## Hidden Layer 3
  with tf.name_scope('fc1'):
    # print(a2)
    # For 28x28:
    # a2flat = tf.reshape(a2, (-1,4*4*16))
    # For 144x256:
    a2flat = tf.reshape(a2, (-1,33*61*16)) 
    # For 128x224:
    # a2flat = tf.reshape(a2, (-1,29*53*16)) 
    # print(a2flat)
    Z3 = 120
    # allocate variables
    # W_fc1 = tf.Variable(npr.uniform(-0.01,0.01, [4*4*16,Z3]),dtype=tf.float32, name ="fc1/weights")
    W_fc1 = tf.Variable(npr.uniform(-0.01,0.01, [33*61*16,Z3]),dtype=tf.float32, name ="fc1/weights")
    b_fc1 = tf.Variable(npr.uniform(-0.01,0.01, [1,Z3]),dtype=tf.float32, name ="fc1/bias")
    # compute activations
    # linear = tf.nn.xw_plus_b(a2flat, W_fc1, b_fc1, name='fc1/linear')
    fc1 = tf.nn.relu(tf.matmul(a2flat, W_fc1) + b_fc1)
    tf.summary.histogram('fc1', fc1)
    tf.summary.histogram('fc1/sparsity', tf.nn.zero_fraction(fc1))
    # print(fc1)
  
  dropout1 = tf.nn.dropout(fc1,0.6,name='dropout1')
  
  # ## Hidden Layer 4
  with tf.name_scope('fc2'):
    Z4 = 84
    # allocate variables
    W_fc2 = tf.Variable(npr.uniform(-0.01,0.01, [Z3,Z4]),dtype=tf.float32, name ="fc2/weights")
    b_fc2 = tf.Variable(npr.uniform(-0.01,0.01, [1,Z4]),dtype=tf.float32, name ="fc2/bias")
    # compute activations
    # linear = tf.nn.xw_plus_b(fc1, W_fc2, b_fc2, name='fc2/linear') 
    fc2 = tf.nn.relu(tf.matmul(fc1, W_fc2) + b_fc2)
    tf.summary.histogram('fc2', fc2)
    tf.summary.histogram('fc2/sparsity', tf.nn.zero_fraction(fc2))
  # print(fc2)

  # dropout2 = tf.nn.dropout(fc2,0.2,name='dropout2')

  # ## Output layer
  with tf.name_scope('output_layer'):
    # alloc variables
    Z5 = totalClasses
    W_out = tf.Variable(npr.uniform(-0.1,0.1, [Z4,Z5]),dtype=tf.float32, name ="out/weights")
    b_out = tf.Variable(npr.uniform(-0.01,0.01, [1,Z5]),dtype=tf.float32, name ="out/bias")
    # compute activations
    logits = tf.matmul(fc2, W_out) + b_out
  # print(logits)

  # ## Loss and Accuracy functions
  with tf.name_scope('cross_entropy_loss'):
    lossBySample = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=label_placeholder)
  # tf.summary.scalar('cross_entropy', tf.reshape(lossBySample,[]) )
  # with tf.name_scope('loss'):
    loss = tf.reduce_mean(lossBySample, name="loss")
    tf.summary.scalar('cross_entropy_loss', loss)

  with tf.name_scope('accuracy'):
    nrCorrect = tf.reduce_mean(tf.cast(tf.equal (tf.argmax(logits,axis=1), tf.argmax(label_placeholder,axis=1)), tf.float32), name='accuracy')
    tf.summary.scalar('accuracy', nrCorrect)

  # ## Create update optimizer
  with tf.name_scope('train'):
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate = LR, name='GradientDescentOptimizer')
    optimizer = tf.train.AdamOptimizer(learning_rate = LR, name='AdamOptimizer')
    update = optimizer.minimize(loss)
  return (logits, update, nrCorrect, loss, data_placeholder, label_placeholder)

# shuffled_img_paths=np.load(data_path, allow_pickle=True)
#To load from pickle file
shuffled_img_paths = []
with open(data_path, 'rb') as fr:
    try:
        while True:
            shuffled_img_paths.append(pickle.load(fr))
    except EOFError:
        pass
# shuffled_img_paths=shuffled_img_paths[:17000]

batch_img_paths = []
for x in batch(shuffled_img_paths, BATCH_SIZE):
    batch_img_paths.append(x)

twenty_percent = int(len(batch_img_paths)*0.2)
print(twenty_percent)
test_img_points_paths = batch_img_paths[:twenty_percent]
train_img_points_paths = batch_img_paths[twenty_percent:]

# Logging Configuration
f=open(result, "a+")
f.write(f"\nShuffled data path = {data_path}\n")
f.write(f"No. of points data = {len(shuffled_img_paths)}\n")
f.write(f"No. of classes = {totalClasses}\n")
f.write(f"Optimizer = AdamOptimizer\n")
# f.write(f"Optimizer = GradientDescentOptimizer\n")
f.write(f"Loss = softmax_cross_entropy_with_logits_v2\n")
f.write(f"LR = {LR}\n")
f.write(f"BATCH_SIZE = {BATCH_SIZE}\n")
f.write(f"EPOCHS = {EPOCHS}\n")
f.close()

# Freeing Memory
shuffled_img_paths = None
batch_img_paths = None

batch_x = []
batch_y = []
logits, update, nrCorrect, loss, data_placeholder, label_placeholder = getModel()
with tf.Session() as sess:
    ## init all variables
    sess.run(tf.global_variables_initializer())
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    # Log info for Tensorboard
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir + '/train-'+datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), sess.graph)
    for epoch in range(EPOCHS):
      for i in range(len(train_img_points_paths)-1):
          batch_x = []
          batch_y = []
          for j in range(BATCH_SIZE):
              point_coordinate, npLabelsHotVector, img_path = train_img_points_paths[i][j]
              img = np.array(Image.open(img_path))
              img = img / 255.0                                             # Normalization and Flattening/Reshaping
              img[imgHeight-1, imgWidth-1] = list(point_coordinate)         # Put pixel coordinates in last pixel of image
              batch_x.append(img)
              batch_y.append(npLabelsHotVector)

          trainFd = {data_placeholder: batch_x, label_placeholder: batch_y}
          sess.run(update, feed_dict=trainFd)                               # Update parameters
          summary, correct, lossVal = sess.run([merged, nrCorrect,loss], feed_dict=trainFd)
          if i%4==0:
            train_writer.add_summary(summary, i)
            print('Epoch {}, acc={:.6f}, loss={:.6f} {}\r'.format(epoch, float(correct), lossVal, i), end='')
            #testacc = sess.run(nrCorrect, feed_dict = testFd)
      print()
      # Save the variables to disk.
      save_path = saver.save(sess, save_directory+'/model.ckpt')
      print("Model saved in path: %s" % save_path)
      f=open(result, "a+")
      f.write('Epoch {}, acc={:.6f}, loss={:.6f}, time={}\r\n'.format(epoch, float(correct), lossVal, time.ctime()))
      f.close()