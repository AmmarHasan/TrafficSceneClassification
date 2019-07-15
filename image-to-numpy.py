import numpy as np;
from PIL import Image;
from pathlib import Path
import _pickle as cPickle
import os

def load_image(filename) :
    img = Image.open(filename)
    img.load()
    # convert the image to grayscale
    img = img.convert(mode='L')
    data = np.asarray(img, dtype="float32")
    return data

counter = 0
imgAsArray = []
dataDirectory = 'ImageCapture'

for i in range(0, len(os.listdir(dataDirectory))):
  filePath = dataDirectory + "/" + str(i) + ".jpg"
  counter+=1
  imgAsArray.append(load_image(filePath))
  print(filePath)


imagesAsArrayFile = 'imagesAsArray'
np.save(imagesAsArrayFile,imgAsArray)
