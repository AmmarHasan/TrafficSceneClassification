from pathlib import Path
from PIL import Image
import numpy as np
import os
import random
import numpy.random as npr
from keras.utils import to_categorical

def createDirectory(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)
  return directory

# Config
px,py,imageType = (256, 144, 'png')
full_res_image_path = 'Data/ScreenCapture-3'
annotationDirectory = full_res_image_path + '/images_segmented-'+ str(px)+'x'+str(py)+'-'+imageType  # -224x128-png/'
trainDirectory = full_res_image_path + '/images_train-'+ str(px)+'x'+str(py)+'-'+imageType  # -224x128-png/'
data_path = full_res_image_path + '/shuffled_img_pathsasd.npy'
verticalPoints = range(int(py*0.35), py-1) # Ignoring upper half of images
horizontalPoints = range(0, px-1)
allCoordinates = [(y,x) for x in horizontalPoints for y in verticalPoints]
segmented_dict = {
  (   0,   0, 255): 'outside', # blue,glitch
  (   0,   0,   0): 'outside', # sky
  (   0, 255, 255): 'outside', # terrrain
  ( 255, 255,   0): 'outside', # tree
  ( 255,   0, 255): 'outside', # house
  ( 255, 153, 153): 'mycar', # mycar
  ( 255, 255, 255): 'otherCar', # otherCar
  ( 128, 128,  76): 'lane4',
  (  76,  76,  76): 'lane3',
  ( 128, 128, 128): 'lane2',
  ( 153, 255, 153): 'lane1',
}
classes = [ 'outside', 'lane1', 'lane2', 'lane3', 'lane4', ]
all_random_points = []
for filePath in Path(annotationDirectory).glob('*' + imageType):
  segmented_pixels = {
    'outside': [],
    'mycar': [],
    'otherCar': [],
    'lane4': [],
    'lane3': [],
    'lane2': [],
    'lane1': [],
  }
  imageSegemented = np.array(Image.open(filePath))
  # pointsData = [(coordinate,segmented_dict[tuple(imageSegemented[coordinate])]) for coordinate in allCoordinates]
  # print(pointsData)
  for coordinate in allCoordinates:
    rgb_key = tuple(imageSegemented[coordinate])
    pixel_class = segmented_dict[rgb_key]
    # Ignoring Car's pixels
    if pixel_class=='otherCar' or pixel_class=='mycar':
      continue
    npLabels = classes.index(pixel_class)
    npLabelsHotVector = to_categorical(npLabels, num_classes=len(classes))
    segmented_pixels[pixel_class].append([coordinate+(0,), npLabelsHotVector.tolist(), str(filePath)])

  random_points = []
  points_count = 0
  for segment in segmented_pixels:
    no_of_segmented_pixels = len(segmented_pixels[segment])
    counter = 0
    print(segment, no_of_segmented_pixels)
    while (no_of_segmented_pixels>0 and counter<no_of_segmented_pixels and counter<400):
      counter+=1
      points_count+=1
      random_index = random.randint(0,no_of_segmented_pixels-1)
      # print(segmented_dict[segment],counter, no_of_segmented_pixels, random_index)
      # print(segmented_pixels[segment][random_index])
      all_random_points.append(segmented_pixels[segment][random_index])

npr.shuffle(all_random_points)
print("shuffled")
np.save(data_path, all_random_points)