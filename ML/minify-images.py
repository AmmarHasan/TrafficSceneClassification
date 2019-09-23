from pathlib import Path
from PIL import Image
import os

def createDirectory(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)
  return directory

def resizeSaveImage(src_path, dest_path, image_name) :
    im = Image.open(src_path)
    nx,ny = im.size
    # Resize images from 1280x720 to 256x144
    # im = im.resize((int(nx/5),int(ny/5)))
    im = im.resize((224,128))
    im.save(dest_path + '/' + image_name +'.jpg')

full_res_image_path = 'Data/ScreenCapture1'
low_res_segmented_image_path = createDirectory('Data/images_segmented-1/')
low_res_image_path = createDirectory('Data/images_train-1/')

counter = 0
for file_path in Path(full_res_image_path).glob('**/**/*_img.jpg'):
    label = file_path.parts[2][-1:]
    image_name = str(counter) + "-" + label
    seg_image_path = str(file_path.parent) + "/" + file_path.name[:-7] + "layer.jpg"

    resizeSaveImage(file_path, low_res_image_path, image_name)
    resizeSaveImage(seg_image_path, low_res_segmented_image_path, image_name)

    counter+=1
    print(counter)