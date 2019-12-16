# ## Initialize data tensor in NHWC format
with tf.name_scope("dataset_inputs") as scope:
  data_placeholder = tf.placeholder(tf.float32,[None,imgHeight, imgWidth,3])
  label_placeholder = tf.placeholder(tf.float32,[None,totalClasses])