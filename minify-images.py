from pathlib import Path
from PIL import Image

counter = 0
for filePath in Path('Data').glob('**/**/*.jpg'):
    im = Image.open(filePath)
    nx,ny = im.size
    # Resize images from 1280x720 to 256x144
    im.resize((int(nx/5),int(ny/5))).save('TransformedData/'+counter.__str__()+'.jpg')
    counter+=1
    print(counter)

