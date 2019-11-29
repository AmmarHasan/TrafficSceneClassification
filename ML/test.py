# # Add pixel coordinates to  fully connected layer tensorflow low-level API
import numpy as np
import numpy.random as npr
from keras.utils import to_categorical
from pathlib import Path
import tensorflow as tf
from PIL import Image
import time

# Config
imgWidth,imgHeight,imageType = (256, 144, 'png')
full_res_image_path = 'Data/ScreenCapture-3'
trainDirectory = full_res_image_path+'/images_train-'+str(imgWidth)+'x'+str(imgHeight)+'-'+imageType
data_path = trainDirectory + '/shuffled_img_paths-test.npy'
# data_path = trainDirectory + '/temp-3.npy'
result='result.txt'
save_directory = full_res_image_path+'/tmp-test-adam'
BATCH_SIZE = 500

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

shuffled_img_paths=np.load(data_path, allow_pickle=True)

batch_img_paths = []
for x in batch(shuffled_img_paths, BATCH_SIZE):
    batch_img_paths.append(x)

twenty_percent = int(len(batch_img_paths)*0.2)
print(twenty_percent)
test_img_points_paths = batch_img_paths[:twenty_percent]

# Freeing Memory
shuffled_img_paths = None
batch_img_paths = None

with tf.Session() as sess:
# with tf.Session() as sess:
    saver = tf.train.import_meta_graph(save_directory+'/model.ckpt.meta')
    graph = tf.get_default_graph()
    data_placeholder = graph.get_tensor_by_name("Images:0")
    label_placeholder = graph.get_tensor_by_name("Labels:0")
    loss = graph.get_tensor_by_name("Mean:0")
    nrCorrect = graph.get_tensor_by_name("accuracy:0")
    total_accuracy = 0
    total_loss = 0
    for i in range(len(test_img_points_paths)):
        batch_x = []
        batch_y = []
        for j in range(BATCH_SIZE):
            point_coordinate, npLabelsHotVector, img_path = test_img_points_paths[i][j]
            img = np.array(Image.open(img_path))
            img = img / 255.0                                             # Normalization and Flattening/Reshaping
            img[imgHeight-1, imgWidth-1] = list(point_coordinate)         # Put pixel coordinates in last pixel of image
            batch_x.append(img)
            batch_y.append(npLabelsHotVector)
        trainFd = {data_placeholder: batch_x, label_placeholder: batch_y}
        saver.restore(sess,tf.train.latest_checkpoint(savedModel))
        # result = sess.run("add_2:0", feed_dict=trainFd)
        # resultClass = np.argmax(result, axis=1)
        # batch_y = np.argmax(batch_y, axis=1)
        # print("resultClass",resultClass)
        # print("batch_y",batch_y)
        # incorrect = len([(i,j) for i, j in zip(resultClass, batch_y) if i != j])
        # correct = BATCH_SIZE - incorrect
        # total_accuracy += (correct )
        # total_loss += (incorrect )
        # print("correct: ", correct)
        # print("incorrect: ", incorrect)
        accuracy = sess.run(nrCorrect, feed_dict=trainFd)
        total_accuracy += (accuracy * len(batch_x))
        print("accuracy: ", accuracy)

    avg_accuracy = total_accuracy / (len(test_img_points_paths)*BATCH_SIZE)
    print("avg_accuracy: ", avg_accuracy)
    print("\n\n")
        
    #     # f=open(result, "a+")
    #     # result = 'Average Accuracy={:.6f}, Average Loss={:.6f}\r'.format(avg_accuracy, avg_loss)
    #     # print(result, end='')
    #     # f.write(result)
    #     # f.write(time.ctime())
    #     # f.close()

    #     # f=open(result, "a+")
    #     # f.write('Epoch {}, acc={:.6f}, loss={:.6f}, time={}\r\n'.format(epoch, float(correct), lossVal, time.ctime()))
    #     # save_path = saver.save(sess, save_directory)
    #     # print("Model saved in path: %s" % save_path)
    #     # f.close()
        
    # print()
