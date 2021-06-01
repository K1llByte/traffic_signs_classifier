################################### Imports ###################################

import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, Conv2D, MaxPooling2D, Dense, Activation, Flatten, Dropout

import os
import pathlib
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
    img = tf.image.decode_image(img, channels=3) # decode_png
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    img = tf.ensure_shape(img, [None, None, 3])

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

    train_listset = tf.data.Dataset.list_files(f"{path}train_images/*/*.*")
    print(type(train_listset))
    train_set = train_listset.map(get_bytes_and_label, num_parallel_calls = AUTOTUNE)

    val_listset = tf.data.Dataset.list_files(f"{path}val_images/*/*.*")
    val_set = val_listset.map(get_bytes_and_label, num_parallel_calls = AUTOTUNE)

    test_listset = tf.data.Dataset.list_files(f"{path}test_images/*/*.*")
    test_set = test_listset.map(get_bytes_and_label, num_parallel_calls = AUTOTUNE)

    print("AAAAAAAAAAAAAAAAAAAAAAA",train_set)

    tmp = [i for i,_ in enumerate(train_set)]
    trainset_length = tmp[-1] + 1
    # trainset_length = [i for i,_ in enumerate(train_set)][-1] + 1

    ############################### Prepare Dataset ################################

    train_set = train_set.cache()
    train_set = train_set.shuffle() # buffer_size=10200
    train_set = train_set.batch(batch_size=BATCH_SIZE)
    train_set = train_set.prefetch(buffer_size=AUTOTUNE)
    train_set = train_set.repeat()

    val_set = val_set.cache()
    val_set = val_set.shuffle() # buffer_size = 2580
    val_set = val_set.batch(batch_size = BATCH_SIZE)
    val_set = val_set.prefetch(buffer_size = AUTOTUNE)
    val_set = val_set.repeat()

    test_set = test_set.batch(batch_size = BATCH_SIZE)


    testset_length = [i for i,_ in enumerate(test_set)][-1] + 1
    #print('Number of batches: ', testset_length)
    return train_set, val_set, test_set, trainset_length

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

def train(in_model, data, model_file='models/cnn1'):
    train_set, val_set, test_set, trainset_length = data
    if not os.path.exists(model_file):
        print("[INFO] Training Model ...")
        in_model.fit(train_set,
                    steps_per_epoch=trainset_length/BATCH_SIZE,
                    epochs=20, 
                    validation_data=val_set, 
                    validation_steps=2580/BATCH_SIZE)
        print("[INFO] Training Finished")

        eval_model = in_model.evaluate(test_set, verbose=1)
        print(eval_model)
        in_model.save(model_file)
        
    else:
        in_model = tf.keras.models.load_model(model_file)
        print("[INFO] Loaded Trained Model")
        in_model.evaluate(test_set, verbose=1)
    return in_model

###############################################################################

def load_and_predict(in_model, img_path):
    img = Image.open(img_path).resize((IMAGE_SIZE,IMAGE_SIZE))
    numpy_image = np.asarray(img)

    # Expand to include batch info
    numpy_image = np.expand_dims(numpy_image, axis=0)

    # Predict
    pred = in_model.predict(numpy_image)
    return class_names[np.argmax(pred[0])]