from pathlib import Path
from PIL import Image
import numpy as np
import os

def createDirectory(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)
  return directory

annotationDirectory = createDirectory('Data/images_segmented/')
trainDirectory = createDirectory('Data/images_train/')
px,py = (224, 128)
pointsTrainingDirectory = createDirectory('Data/points_train/')
pointsLabelDirectory = createDirectory('Data/points_label/')
imageType = '.png'
verticalPoints = range(int(py*0.4), py, int((py-(py*0.4))/30))
horizontalPoints = range(0, px, int(px/70))
points = [(y,x) for x in horizontalPoints for y in verticalPoints]
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

for filePath in Path(trainDirectory).glob('*' + imageType):
  label = str(int(filePath.parts[-1][:-4]))
  segmentedImagePath = annotationDirectory+label+imageType
  image = np.array(Image.open(filePath))
  imageSegemented = np.array(Image.open(segmentedImagePath))
  pointsTrainData = [(point,image[point]) for point in points]
  pointsLabelData = [segmented_dict[tuple(imageSegemented[point])] for point in points]
  np.save(createDirectory(trainDirectory+'numpy/')+label+'.npy', image)
  np.save(createDirectory(annotationDirectory+'numpy/')+label+'.npy', imageSegemented)
  np.save(pointsTrainingDirectory+label+'.npy', pointsTrainData)
  np.save(pointsLabelDirectory+label+'.npy', pointsLabelData)
  