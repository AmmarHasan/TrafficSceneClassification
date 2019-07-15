from pathlib import Path
from PIL import Image
import numpy as np
import _pickle as cPickle

def load_image(img) :
    # convert the image to grayscale
    img = img.convert(mode='L')
    data = np.asarray(img, dtype="float32")
    return data

fullResImageDirectory = 'Data'
lowResImageDirectory = 'ImageCapture'
labelsFileName = 'labelsAll.data'
imagesAsArrayFile = 'imagesAsArray.npy'
noOfLanes = 4

counter = 0
labels = []
imgAsArray = []

for filePath in Path(fullResImageDirectory).glob('**/**/*.jpg'):
    im = Image.open(filePath)
    nx,ny = im.size
    # Resize images from 1280x720 to 256x144
    im = im.resize((int(nx/5),int(ny/5)))
    im.save(lowResImageDirectory + '/' + counter.__str__()+'.jpg')
    imgAsArray.append(load_image(im))

    counter+=1
    label = int(filePath.parts[1][-1:])
    print(counter)
    print("label:" + str(label))
    labels.append(label)

# Generating one hot vector
npLabels = np.array(labels)
npLabelsHotVector = np.zeros((npLabels.size, noOfLanes))
npLabelsHotVector[np.arange(npLabels.size), npLabels-1] = 1

print(npLabels.shape)
np.save(labelsFileName, npLabels)
npLabels = None
print(npLabelsHotVector.shape)
np.save("labelsVector.npy", npLabelsHotVector)


print("writing image array")
np.save(imagesAsArrayFile, imgAsArray)