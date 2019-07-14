from pathlib import Path
from PIL import Image
import numpy as np
import _pickle as cPickle

noOfLanes = 4
labels = []
labelsFileName = 'labels.data'
counter = 0
fullResImageDirectory = 'Data'
lowResImageDirectory = 'ImageCapture'
for filePath in Path(fullResImageDirectory).glob('**/**/*.jpg'):
    im = Image.open(filePath)
    nx,ny = im.size
    # Resize images from 1280x720 to 256x144
    im.resize((int(nx/5),int(ny/5))).save(lowResImageDirectory + '/' + counter.__str__()+'.jpg')
    counter+=1
    label = int(filePath.parts[1][-1:])
    print(counter)
    print("label:" + str(label))
    labels.append(label)

# Generating one hot vector
npLabels = np.array(labels)
npLabelsHotVector = np.zeros((npLabels.size, noOfLanes))
npLabelsHotVector[np.arange(npLabels.size), npLabels-1] = 1


with open(lowResImageDirectory + '/' + labelsFileName, 'wb') as filehandle:
    # store the data as binary data stream
    cPickle.dump(npLabelsHotVector, filehandle)

print(npLabelsHotVector.argmax(axis=0))