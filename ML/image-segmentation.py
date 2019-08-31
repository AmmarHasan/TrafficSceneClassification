# from keras.backend import tensorflow_backend
import keras_segmentation

# model = keras_segmentation.pretrained.pspnet_101_cityscapes()
# out = model.predict_segmentation(
#     inp="img.png",
#     out_fname="out.png"
# )

model = keras_segmentation.models.unet.unet(n_classes=6, input_height=128, input_width=224)
model.train( 
    train_images =  "Data/images_prepped_train_vgg/",
    train_annotations = "Data/annotations_prepped_train_vgg/",
    checkpoints_path = "/tmp/vgg_unet_1" , epochs=5
)
out = model.predict_segmentation(
    inp="Data/images_prepped_test/1.png",
    out_fname="/tmp/out.png"
)

import matplotlib.pyplot as plt
plt.imshow(out)
