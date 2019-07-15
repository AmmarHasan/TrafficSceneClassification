from PIL import Image;
import os

dataDirectory = 'ImageCapture'
grayscaleDataDirectory = 'ImageCaptureGrayscale'

for i in range(0, len(os.listdir(dataDirectory))):
  filename = str(i) + '.jpg'
  filePath = dataDirectory + '/' + filename
  destFilePath = grayscaleDataDirectory + '/' + filename
  img = Image.open(filePath).convert(mode='L')
  img.save(destFilePath)
  print(destFilePath)

