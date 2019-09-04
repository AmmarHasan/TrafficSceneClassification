from pathlib import Path
from PIL import Image
import numpy as np

annotationDirectory = 'Data/images_segmented/'
trainDirectory = 'Data/images_train/'
px,py = (1280, 720)
pointsTrainingDirectory = 'Data/points_train/'
pointsLabelDirectory = 'Data/points_label/'
imageType = '.png'
verticalPoints = range(int(py*0.4), py, int((py-(py*0.4))/30))
horizontalPoints = range(0, px, int(px/70))
points = [(y,x) for x in horizontalPoints for y in verticalPoints]

for filePath in Path(trainDirectory).glob('*' + imageType):
  label = int(filePath.parts[-1][:-4])
  segmentedImagePath = annotationDirectory + str(label) + imageType
  image = np.array(Image.open(filePath))
  imageSegemented = np.array(Image.open(segmentedImagePath))
  pointsTrainData = [(point,image[point]) for point in points]
  pointsLabelData = [(point,imageSegemented[point]) for point in points]
  np.save(pointsTrainingDirectory+str(label)+'.npy', pointsTrainData)
  np.save(pointsLabelDirectory+str(label)+'.npy', pointsLabelData)
