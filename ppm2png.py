from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
import glob
# print(len(glob.glob("basement_0001a/*.pgm")))
dirs = glob.glob("./DATASET/*/")

for d in dirs:
    # print(d)

    subdirs = glob.glob(d+"*/")
    for s in subdirs:
        print(s)
        xfiles = glob.glob(s+"*.ppm")
        # print(xfiles)
        num = 0
        pngsubdir = './png/'+s[2:-1]
        print(pngsubdir)
        if not os.path.exists(pngsubdir):
            os.makedirs(pngsubdir)

        for f in xfiles:
            im = Image.open(f)
            im.save("png/"+s+str(num)+".png")
            num +=1

# for d in glob.glob("./*/"):
#     print(glob.glob(d+"/*"))
import matplotlib.pyplot as plt
# im = plt.imread("png/r-1316653687.842499-3408451313.png", format=None)

# plt.imshow(im, cmap='gray')
# plt.show()