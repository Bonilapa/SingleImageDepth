import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

'''
Data augmentation
'''
generator = ImageDataGenerator(rotation_range=5,
                               zoom_range=[1.0, 1.5],
                               horizontal_flip=True,
                               validation_split=0.15,
                               )

generator.fit(xdata)


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


from keras.callbacks import EarlyStopping, ReduceLROnPlateau

model = make_model()
model.compile(optimizer=SGD(1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(generator.flow(xdata,
                         ydata,
                         batch_size=128),
          validation_data=generator.flow(xdata,
                                         ydata,
                                         batch_size=128,
                                         subset='validation'),
          steps_per_epoch=len(xdata) / 128,
          epochs=300,
          verbose=2,
          batch_size=128, )
print(model.evaluate(xtest))
model.save('model.h5')