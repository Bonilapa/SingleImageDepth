from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
import glob
import re
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

# print(len(glob.glob("basement_0001a/*.pgm")))
dirs = glob.glob("./DATASET/*/")

for d in dirs:
    # print(d)

    subdirs = glob.glob(d+"*/")
    for s in subdirs:
        # print(s)
        xfiles = glob.glob(s+"*.pgm")
        xfiles.sort(key=natural_keys)
        # print(xfiles)
        pngsubdir = './png/depth'+s[9:-1]
        print(pngsubdir)
        if not os.path.exists(pngsubdir):
            os.makedirs(pngsubdir)
            print("created\n")

            num = 0
            for f in xfiles:
                try:
                    im = Image.open(f)
                except ValueError:
                    print("Oops!  That was no valid number.  Try again...")
                    continue
                im.save(pngsubdir+"/"+str(num)+".png")
                num +=1

# for d in glob.glob("./*/"):
#     print(glob.glob(d+"/*"))
import matplotlib.pyplot as plt
# im = plt.imread("png/r-1316653687.842499-3408451313.png", format=None)

# plt.imshow(im, cmap='gray')
# plt.show()