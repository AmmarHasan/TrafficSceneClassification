from pathlib import Path
from PIL import Image
import numpy as np
import os
import random
import numpy.random as npr

def createDirectory(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)
  return directory

annotationDirectory = createDirectory('Data/images_segmented-224x128-png/')
trainDirectory = createDirectory('Data/images_train-224x128-png/')
px,py = (224, 128)
pointsTrainingDirectory = createDirectory('Data/points_train/')
pointsLabelDirectory = createDirectory('Data/points_label/')
imageType = '.png'
verticalPoints = range(0, py-1)
horizontalPoints = range(0, px-1)
allCoordinates = [(y,x) for x in horizontalPoints for y in verticalPoints]
segmented_dict = {
  (  0,   0,   0): 'sky',
  (  0, 255, 255): 'terrrain',
  (255, 255,   0): 'tree',
  (255,   0, 255): 'house',
  (255, 178, 178): 'mycar',
  (255, 255, 255): 'otherCar',
  (178, 255, 178): 'lane4',
  (128, 128, 128): 'lane3',
  ( 89,  89,  89): 'lane2',
  (  0,   0, 255): 'lane1',
}

for filePath in Path(annotationDirectory).glob('*' + imageType):
  segmented_pixels = {
    (  0,   0,   0): [],
    (  0, 255, 255): [],
    (255, 255,   0): [],
    (255,   0, 255): [],
    (255, 178, 178): [],
    (255, 255, 255): [],
    (178, 255, 178): [],
    (128, 128, 128): [],
    ( 89,  89,  89): [],
    (  0,   0, 255): [],
  }
  imageSegemented = np.array(Image.open(filePath))
  # pointsData = [(coordinate,segmented_dict[tuple(imageSegemented[coordinate])]) for coordinate in allCoordinates]
  # print(pointsData)
  for coordinate in allCoordinates:
    rgb_key = tuple(imageSegemented[coordinate])
    segmented_pixels[rgb_key].append([coordinate+(0,),segmented_dict[rgb_key]])

  random_points = []
  for segment in segmented_pixels:
    no_of_segmented_pixels = len(segmented_pixels[segment])
    counter = 0
    while (no_of_segmented_pixels>0 and counter<no_of_segmented_pixels and counter<300):
      counter+=1
      random_index = random.randint(0,no_of_segmented_pixels-1)
      # print(segmented_dict[segment],counter, no_of_segmented_pixels, random_index)
      # print(segmented_pixels[segment][random_index])
      random_points.append(segmented_pixels[segment][random_index])

  label = filePath.parts[-1][:-4]
  npr.shuffle(random_points)
  np.save(createDirectory(trainDirectory+'points/')+label+'.npy', random_points[0:1000])
  print(trainDirectory+'points/'+label+'.npy')
  
# a=np.load(trainDirectory+'points/1509-2.npy', allow_pickle=True)
# print(a)
# print(a.shape)