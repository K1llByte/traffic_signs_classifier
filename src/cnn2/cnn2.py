from tensorflow.keras.models import Model, Sequential, model_from_json
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, Conv2D, MaxPooling2D, Dense, Activation, Flatten, Dropout
from tensorflow.keras.regularizers import l2
#import tensorflow.keras.layers as kl
from tensorflow.keras.layers import *
import tensorflow as tf
import os
import math

#BATCH_SIZE = 32
BATCH_SIZE = 50
IMAGE_SIZE = 32



#################################### Model ####################################

################################ Define Model #################################

def make_model(class_count, img_size, channels=3):

    model = Sequential()

    model.add(Conv2D(1, (1,1), padding='same', input_shape=(img_size, img_size, channels)))
    model.add(BatchNormalization())
    model.add(ReLU()) # LeakyReLU(alpha=0.01)

    model.add(Conv2D(29, (5,5)))
    model.add(BatchNormalization())
    model.add(ReLU()) # LeakyReLU(alpha=0.01)
    model.add(MaxPooling2D(pool_size=3, strides=2)) # apenas 3 basta e fica 3x3

    model.add(Conv2D(59, (3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU()) # LeakyReLU(alpha=0.01)
    model.add(MaxPooling2D(pool_size=3, strides=2)) # apenas 3 basta e fica 3x3

    model.add(Conv2D(74, (3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU()) # LeakyReLU(alpha=0.01)
    model.add(MaxPooling2D(pool_size=3, strides=2)) # apenas 3 basta e fica 3x3

    model.add(Flatten())
    model.add(Dense(300)) # LeakyReLU(alpha=0.01)
    model.add(BatchNormalization())
    model.add(ReLU()) # LeakyReLU(alpha=0.01)
    model.add(Dense(300))
    model.add(ReLU()) # LeakyReLU(alpha=0.01)
    model.add(Dense(class_count)) # ou alternativamente , activation='softmax'
    model.add(Softmax())


    opt = SGD(learning_rate=0.0007, momentum=0.9, nesterov=True)

    model.compile(optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model

def train(in_model, data, model_file='models/cnn2',num_epochs=20, evalu=True):
    train_set, val_set, test_set, dataset_length = data
    steps = math.ceil(dataset_length * 0.3)/BATCH_SIZE
    if not os.path.exists(model_file):
        print("[INFO] Training Model ...")
        in_model.fit(train_set,
                    steps_per_epoch=dataset_length/BATCH_SIZE,
                    epochs=num_epochs,
                    validation_data=val_set, 
                    validation_steps=steps)
        print("[INFO] Training Finished")

        if evalu:
            print("Test Set:")
            values = in_model.evaluate(test_set, verbose=1)
            for metric, val in zip(in_model.metrics_names,values):
                print(f'{metric}: {val}')
        in_model.save(model_file)

    else:
        in_model = tf.keras.models.load_model(model_file)
        print("[INFO] Loaded Trained Model")
        if evalu:
            print("Test Set:")
            values = in_model.evaluate(test_set, verbose=1)
            for metric, val in zip(in_model.metrics_names,values):
                print(f'{metric}: {val}')
    return in_model