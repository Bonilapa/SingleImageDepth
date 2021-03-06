import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import re

from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

def read_and_resize(filename: str, grayscale: bool = False, fx: float = 1.0, fy: float = 1.0):
    if grayscale:
        img_result = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    else:
        imgbgr = cv2.imread(filename, cv2.IMREAD_COLOR)
        # convert to rgb
        img_result = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2RGB)
    # resize
    if fx != 1.0 and fy != 1.0:
        img_result = cv2.resize(img_result, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
    return img_result


def show_in_row(list_of_images: list, titles: list = None, disable_ticks: bool = False):
    count = len(list_of_images)
    for idx in range(count):
        subplot = plt.subplot(1, count, idx + 1)
        if titles is not None:
            subplot.set_title(titles[idx])

        img = list_of_images[idx]
        cmap = 'gray' if (len(img.shape) == 2 or img.shape[2] == 1) else None
        subplot.imshow(img, cmap=cmap)
        if disable_ticks:
            plt.xticks([]), plt.yticks([])
    plt.show()


dirs = glob.glob("./png/DATASET/*/")

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def make_model():
    coarse = Sequential()
    coarse.add(Conv2D(filters=96, kernel_size=15, stride=5, input_shape=(320, 240, 3), activation='relu'))
    coarse.add(MaxPool2D(pool_size=2, strides=1))
    coarse.add(Conv2D(filters=128, kernel_size=5, stride=3, activation='relu'))
    coarse.add(Conv2D(filters=128, kernel_size=3, stride=1, activation='relu'))
    coarse.add(Conv2D(filters=64, kernel_size=3, stride=1, activation='relu'))

    coarse.add(Flatten())
    coarse.add(Dense(5120, activation='lineae'))
    coarse.add(Dense(4800, activation=''))

    fine = Sequential()
    fine.add(Conv2D(filters=59, kernel_size=5, stride=2, padding=1, input_shape=(321, 240, 3), activation='relu'))
    # concat

    fine.add(Conv2D(filters=60, kernel_size=3, stride=1, padding=1, input_shape=(321, 240, 3), activation='relu'))
    return fine

model = make_model()

# path = "basements/basement_0001a/"+str(name)+".png"
# color_path = "./png//DATASET/" + path
# depth_path = "./png/depth/" + path

for d in dirs:
    # print(d)

    subdirs = glob.glob(d+"*/")
    for s in subdirs:
        print(s+" reading...")
        xfiles = glob.glob(s+"*.png")
        xfiles.sort(key=natural_keys)
        # print(xfiles)

        # s = s.split("/")[-2]

        count = 0
        ok_count = 0
        color_images = []
        depth_images = []

        for f in xfiles:
            # f = f.split("/")[-1]
            # print(f)


            tmp = f.split("/")
            tmp[2] = "depth"
            d_f = '/'.join(tmp)
            # print("==================================="+d_f)
            # png_depth_file = './png/depth' + s[9:-1]

            # not all color images have depth accordance in original dataset
            if not os.path.exists(d_f):
                # print("_____CAUTION!! file has no pair: ___ "+f)
                count = count + 1
            else:
                ok_count = ok_count +1

                color_images.append(read_and_resize(f, True, fx=0.5, fy=0.5))
                depth_images.append(read_and_resize(d_f, True, fx=0.5, fy=0.5))

            #     try:
            #         im = read_and_resize(f)
            #     except OSError as e:
            #         print(e)
            #         continue
            #     num +=1
        # temporary break the loop only for 1 folder processing on testing stage
        break

        color_images_n = np.array(color_images)
        depth_images_n = np.array(depth_images)
        '''
        Data augmentation
        '''
        generator = ImageDataGenerator(rotation_range=5,
                                       zoom_range=[1.0, 1.5],
                                       horizontal_flip=True,
                                       validation_split=0.15,
                                       )

        generator.fit(color_images)



        '''
        model.fit here
        '''

        model.compile(optimizer=SGD(1e-4),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(generator.flow(color_images_n,
                                 depth_images_n,
                                 batch_size=128),
                  validation_data=generator.flow(color_images_n,
                                                 depth_images_n,
                                                 batch_size=128,
                                                 subset='validation'),
                  steps_per_epoch=len(xdata) / 128,
                  epochs=10,
                  verbose=2,
                  batch_size=128, )
    # temporary break the loop only for 1 folder processing on testing stage
    break
print("Data reading result:")
print(str(count)+" images without depth pair")
print(str(ok_count)+" full pairs")
color_images_n = np.array(color_images)
depth_images_n = np.array(depth_images)
print("X_data:\n"+str(color_images_n.shape))
print("Y_data:\n"+str(depth_images_n.shape))

show_in_row([color_images[0], depth_images[0]])