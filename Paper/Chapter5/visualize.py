# Config
imgWidth,imgHeight,imageType = (256, 144, 'png')
full_res_image_path = 'Data/ScreenCapture-2'
trainDirectory = full_res_image_path+'/images_train-'+str(imgWidth)+'x'+str(imgHeight)+'-'+imageType
# Ignoring upper half of images
verticalPoints = range(int(imgHeight*0.50), imgHeight-30, 2)
horizontalPoints = range(20, imgWidth-30, 2)
allCoordinates = [(y,x) for x in horizontalPoints for y in verticalPoints]
allCoordinates3 = [c+(0,) for c in allCoordinates]
savedModel = 'tmp-test-1'
# black,red,green,blue,white
colors = [[0,0,0],[255,0,0],[0,255,0],[0,0,255],[255,255,255]]
file_names = ['3533-1','307-2','2664-3','3000-3']
img_paths = [trainDirectory+'/'+name+'.png' for name in file_names]

# Putting coordinates of the point to predict in the last pixel
all_batches = []
for img_path in img_paths:
  batch_x = []
  img = np.array(Image.open(img_path))
  img = img / 255.0                                             # Normalization and Flattening/Reshaping
  for i in range(len(allCoordinates3)):
      point_coordinate = allCoordinates3[i]
      imgCopy = copy.copy(img)
      imgCopy[imgHeight-1, imgWidth-1] = list(point_coordinate)  # Put pixel coordinates in last pixel of image
      batch_x.append(imgCopy)
  all_batches.append(batch_x)

with tf.Session() as sess:
  saver = tf.train.import_meta_graph(savedModel+'/model.ckpt.meta')
  graph = tf.get_default_graph()
  data_placeholder = graph.get_tensor_by_name("Image:0")
  for j in range(len(img_paths)):
    trainFd = {data_placeholder: all_batches[j]}
    # Restore saved model which was trained during training process
    saver.restore(sess,tf.train.latest_checkpoint(savedModel))
    result = sess.run("logits:0", feed_dict=trainFd)
    resultClass = np.argmax(result, axis=1)
    image = cv2.imread(img_paths[j])
    for k in range(len(allCoordinates)):
      coordinate = allCoordinates[k]
      # Create a named colour
      color = colors[resultClass[k]]
      # Change one pixel
      image[coordinate[0], coordinate[1]]=color

    image = cv2.resize(image,(4*imgWidth, 4*imgHeight), interpolation = cv2.INTER_CUBIC)
    # Show Visualization
    cv2.imshow(img_paths[j],image)

