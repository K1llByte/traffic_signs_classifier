################################### Imports ###################################

import tensorflow as tf

#from tensorflow import keras
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

############################## Auxiliar Functions #############################

def show_batch(cols, image_batch, label_batch):

    rows = int(BATCH_SIZE / cols) 
    if rows * cols < BATCH_SIZE:
        rows += 1
    width = 3 * rows
    height = 3 * cols
    
    
    f, axes= plt.subplots(rows,cols,figsize=(height,width))
    fig=plt.figure()
    for n in range(BATCH_SIZE):
        
        subplot_title=("class "+ class_names[label_batch[n]==1][0])
        axes.ravel()[n].set_title(subplot_title)  
        axes.ravel()[n].imshow(image_batch[n])
        axes.ravel()[n].axis('off')

    fig.tight_layout()    
    plt.show()
    
    
def show_history(history):
    print(history.history.keys())

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='lower right')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()    
    
    
def show_accuracies(): 
    fig, ax = plt.subplots()
    X = np.arange(2)

    models = ['simple', 'new layers']
    plt.bar(X, [evalV1[1], evalV2[1]], width = 0.4, color = 'b', label='test')
    plt.bar(X + 0.4, [valV1[1], valV2[1]], color = 'r', width = 0.4, label = "val")
    plt.xticks(X + 0.4 / 2, models)
    plt.ylim(top = 1.0, bottom = 0.80)
    plt.legend(loc='upper left')
    plt.show()


def show_data(s1,l1, s2,l2, labels):
    fig, ax = plt.subplots()
    X = np.arange(len(s1))

    models = labels
    plt.bar(X, s1, width = 0.4, color = 'b', label=l1)
    plt.bar(X + 0.4, s2, color = 'r', width = 0.4, label = l2)
    plt.xticks(X + 0.4 / 2, models)
    plt.ylim(top = 100, bottom = 90)
    plt.legend(loc='upper left')
    plt.show()



def show_misclassified(predictions, ground_truth, images, num_rows= 5, num_cols=3):
    
    # Plot the first X test images with wrong predictions.
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    i = 0
    k = 0
    while k < len(images) and i < num_images:
        predicted_label = np.argmax(predictions[k])
        gt = np.where(ground_truth[k])[0][0]
        if predicted_label != gt:
            plt.subplot(num_rows, 2*num_cols, 2*i+1)
            plot_image(k, predictions[k], gt, images)
            plt.subplot(num_rows, 2*num_cols, 2*i+2)
            plot_value_array(k, predictions[k], ground_truth)
            i += 1
        k += 1
    plt.tight_layout()
    plt.show()


def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label, img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(8))
  plt.yticks([])
  thisplot = plt.bar(range(8), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[np.where(true_label)[0][0]].set_color('blue')    

def plot_predictions(predictions, ground_truth, images, num_rows= 5, num_cols=3 ):

    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(min(num_images,len(images))):
        gt = np.where(ground_truth[i])[0][0]
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions[i], gt, images)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions[i], ground_truth)
    plt.tight_layout()
    plt.show()


class_names = np.array(['00000','00001', '00002', '00003', '00004', '00005', '00006', '00007'])

def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [IMAGE_SIZE,IMAGE_SIZE])

def get_bytes_and_label(file_path):
    def get_label(file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        return parts[-2] == class_names

    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label



def fetch_data(path="data/gtsrb"):

    if path[-1] != '/':
        path += '/'

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_listset = tf.data.Dataset.list_files(f"{path}train_images/*/*.png")
    train_set = train_listset.map(get_bytes_and_label, num_parallel_calls = AUTOTUNE)

    val_listset = tf.data.Dataset.list_files(f"{path}val_images/*/*.png")
    val_set = val_listset.map(get_bytes_and_label, num_parallel_calls = AUTOTUNE)

    test_listset = tf.data.Dataset.list_files(f"{path}test_images_per_folder/*/*.png")
    test_set = test_listset.map(get_bytes_and_label, num_parallel_calls = AUTOTUNE)

    dataset_length = [i for i,_ in enumerate(train_set)][-1] + 1

    ################################ Prepare Model ################################

    train_set = train_set.cache()
    train_set = train_set.shuffle(buffer_size=10200)
    train_set = train_set.batch(batch_size=BATCH_SIZE)
    train_set = train_set.prefetch(buffer_size=AUTOTUNE)
    train_set = train_set.repeat()

    val_set = val_set.cache()
    val_set = val_set.shuffle(buffer_size = 2580)
    val_set = val_set.batch(batch_size = BATCH_SIZE)
    val_set = val_set.prefetch(buffer_size = AUTOTUNE)
    val_set = val_set.repeat()

    test_set = test_set.batch(batch_size = BATCH_SIZE)

    testset_length = [i for i,_ in enumerate(test_set)][-1] + 1
    #print('Number of batches: ', testset_length)
    return train_set, val_set, test_set

#################################### Model ####################################

################################ Define Model #################################

def make_model(class_count, img_size, channels):
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

    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# modelV3 = make_model(8, IMAGE_SIZE, 3)

# ################################# Train Model #################################

def train(in_model, data):
    MODEL_FILE = 'models/'
    if not os.path.exists(MODEL_FILE):
        modelV3.fit(train_set,
                    steps_per_epoch=dataset_length/BATCH_SIZE,
                    epochs=20, 
                    validation_data=val_set, 
                    validation_steps=2580/BATCH_SIZE)


        eval = model.evaluate(test_set, verbose=2)
        print(evalV3)
        modelV3.save(MODEL_FILE)
    else:
        modelV3 = tf.keras.models.load_model(MODEL_FILE)

# modelV3.fit(train_set, steps_per_epoch=dataset_length/BATCH_SIZE, epochs=20)


# evalV3 = modelV3.evaluate(test_set, verbose=2)
# print(evalV3)

# ###############################################################################

# im = Image.open("new_50_1.jpg").resize((IMAGE_SIZE,IMAGE_SIZE))
# numpy_image = np.asarray(im)

# # Expand to include batch info
# numpy_image = np.expand_dims(numpy_image, axis=0)

# # Predict
# in_pred = modelV3.predict(numpy_image)
# print(in_pred)