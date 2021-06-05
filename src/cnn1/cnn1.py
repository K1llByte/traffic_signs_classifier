################################### Imports ###################################

import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, Conv2D, MaxPooling2D, Dense, Activation, Flatten, Dropout
import tensorflow_addons as tfa

import os
import pathlib
import math
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import IPython.display as display

BATCH_SIZE = 32
IMAGE_SIZE = 32
#class_ids = np.array(['00000','00001', '00002', '00003', '00004', '00005', '00006', '00007'])
#class_names = ['Limit 20', 'Limit 30', 'Limit 51', 'Limit 60', 'Limit 70', 'Limit 80', 'Limit 100', 'Limit 120']
class_ids = np.array(os.listdir('data/gtsrb_full/train_images'))
class_names = class_ids
NUM_CLASSES = len(class_ids)

############################## Auxiliar Functions #############################

def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    
    #img = tf.ensure_shape(img, [None, None, 3])

    return tf.image.resize(img, [IMAGE_SIZE,IMAGE_SIZE])

def get_bytes_and_label(file_path):
    def get_label(file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        return parts[-2] == class_ids

    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

################################### Dataset ###################################

def fetch_data(path="data/gtsrb_full"):

    ################################ Load Dataset ##################################

    if path[-1] != '/':
        path += '/'

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_listset = tf.data.Dataset.list_files(f"{path}train_images/*/*.png")
    train_set = train_listset.map(get_bytes_and_label, num_parallel_calls = AUTOTUNE)

    val_listset = tf.data.Dataset.list_files(f"{path}val_images/*/*.png")
    val_set = val_listset.map(get_bytes_and_label, num_parallel_calls = AUTOTUNE)

    test_listset = tf.data.Dataset.list_files(f"{path}test_images/*/*.png")
    test_set = test_listset.map(get_bytes_and_label, num_parallel_calls = AUTOTUNE)

    tmp = [i for i,_ in enumerate(train_set)]
    trainset_length = tmp[-1] + 1
    # trainset_length = [i for i,_ in enumerate(train_set)][-1] + 1

    ############################### Prepare Dataset ################################

    train_set_len = tf.data.experimental.cardinality(train_set).numpy()
    val_set_len = tf.data.experimental.cardinality(val_set).numpy()
    
    train_set = train_set.cache()
    train_set = train_set.shuffle(buffer_size=train_set_len)
    train_set = train_set.batch(batch_size=BATCH_SIZE)
    train_set = train_set.prefetch(buffer_size=AUTOTUNE)
    train_set = train_set.repeat()

    val_set = val_set.cache()
    val_set = val_set.shuffle(buffer_size=val_set_len)
    val_set = val_set.batch(batch_size = BATCH_SIZE)
    val_set = val_set.prefetch(buffer_size = AUTOTUNE)
    val_set = val_set.repeat()

    test_set = test_set.batch(batch_size = BATCH_SIZE)


    # testset_length = [i for i,_ in enumerate(test_set)][-1] + 1
    # print('Number of batches: ', testset_length)

    return train_set, val_set, test_set, trainset_length



# Brightness
def process_image_brightness(image, label):
    image = tf.clip_by_value(tf.image.random_brightness(image, max_delta = 0.25), 0, 1)
    return image, label 


# Contrast
def process_image_contrast(image, label):
    image = tf.clip_by_value(tf.image.random_contrast(image, lower=0.7, upper=1.3, seed=None), 0, 1)
    return image

# Saturation
def process_image_saturation(image, label):
    image = tf.image.random_saturation(image, lower=0.6, upper= 1.4, seed=None)
    return image


# Contrast
def process_image_translate(image, label):
    rx = tf.random.uniform(shape=(), minval=-10, maxval=10)
    ry = tf.random.uniform(shape=(), minval=-4, maxval=4) - 4
    image = tfa.image.translate(image, [rx, ry])
    return image, label

# Saturation
def process_image_rotate(image, label):
    r = tf.random.uniform(shape=(), minval=0, maxval=0.5) - 0.25
    image = tfa.image.rotate(image, r)
    #image = tf.clip_by_value(tfa.image.random_hsv_in_yiq(image, 0.0, 0.4, 1.1, 0.4, 1.1), 0.0, 1.0)
    #image = tf.clip_by_value(tf.image.adjust_brightness(image, tf.random.uniform(shape=(), minval=0, maxval=0.1)-0.2),0,1)
    return image, label


def data_augmentation(data):
    def apply_all(image, label):
        image, label = process_image_brightness(image, label)
        image, label = process_image_contrast(image, label)
        image, label = process_image_saturation(image, label)
        image, label = process_image_translate(image, label)
        image, label = process_image_rotate(image, label)
        return image, label

    train_set, val_set, test_set, dataset_length = data

    new_train_set = train_set.map(process_image_translate)
    new_train_set = new_train_set.concatenate(train_set.map(apply_all))
    
    new_train_set = new_train_set.shuffle(20)
    new_train_set = new_train_set.batch(batch_size=BATCH_SIZE)
    new_train_set = new_train_set.repeat()
    return new_train_set, val_set, test_set, dataset_length*2 


#################################### Model ####################################

################################ Define Model #################################

def make_model(class_count, img_size, channels=3):
    model = Sequential()
    
    model.add(Conv2D(64, (5, 5), 
                     input_shape=(img_size, img_size, channels)
                     ))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    
    model.add(Conv2D(128, (5, 5) )) 
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (5, 5)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.2))

    model.add(Dense(class_count, activation='softmax'))

    model.compile(
        optimizer=Adam(lr=0.0001), 
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model

################################# Train Model #################################

def train(in_model, data, model_file='models/cnn1',num_epochs=20, evalu=True):
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

        values = in_model.evaluate(test_set, verbose=1)
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

###############################################################################

