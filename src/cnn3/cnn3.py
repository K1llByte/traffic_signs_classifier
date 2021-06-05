from tensorflow.keras.models import Model, Sequential, model_from_json
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
import tensorflow.keras.layers as kl
import tensorflow as tf
import os
import math

def make_model(class_count, img_size, channels=3):

    model = Sequential()

    model.add(Conv2D(1, (1,1), padding='same', input_shape=(img_size, img_size, channels)))
    model.add(BatchNormalization())
    model.add(ReLU()) # LeakyReLU(alpha=0.01)

    model.add(Conv2D(64, (5, 5) )) 
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3) )) 
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Flatten())
    model.add(Dense(256)) # LeakyReLU(alpha=0.01)
    model.add(BatchNormalization())
    model.add(ReLU()) # LeakyReLU(alpha=0.01)
    model.add(Dropout(0.2))

    model.add(Dense(128))
    model.add(ReLU()) # LeakyReLU(alpha=0.01)
    model.add(Dense(class_count)) # ou alternativamente , activation='softmax'
    model.add(Softmax())


    opt = Adam(lr=0.0001)
    model.compile(optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model