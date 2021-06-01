from tensorflow.keras.models import Model, Sequential, model_from_json
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, Conv2D, MaxPooling2D, Dense, Activation, Flatten, Dropout
from tensorflow.keras.regularizers import l2
import tensorflow.keras.layers as kl
import tensorflow as tf
import os

BATCH_SIZE = 32
IMAGE_SIZE = 32

#################################### Model ####################################

################################ Define Model #################################

def make_model(class_count, img_size, channels=3):
    lr = 0.0007
    l2_reg_rate = 1e-5
    eps = 1e-6

    input_ = kl.Input(shape=(img_size,img_size,channels), name='data')
    # 1 Part
    x = kl.Conv2D(filters=1, kernel_size=(1,1), padding='same',
            kernel_regularizer=l2(l2_reg_rate))(input_)
    x = kl.BatchNormalization(epsilon=eps)(x)
    x = kl.ReLU()(x)
    # 2 Part
    x = kl.Conv2D(filters=29, kernel_size=(5,5),
            kernel_regularizer=l2(l2_reg_rate))(x)
    x = kl.BatchNormalization(epsilon=eps)(x)
    x = kl.ReLU()(x)
    x = kl.MaxPooling2D(pool_size=3, strides=2)(x)
    # 3 Part
    x = kl.Conv2D(filters=59, kernel_size=(3,3), padding='same',
            kernel_regularizer=l2(l2_reg_rate))(x)
    x = kl.BatchNormalization(epsilon=eps)(x)
    x = kl.ReLU()(x)
    x = kl.MaxPooling2D(pool_size=3, strides=2)(x)
    # 4 Part
    x = kl.Conv2D(filters=74, kernel_size=(3,3), padding='same',
            kernel_regularizer=l2(l2_reg_rate))(x)
    x = kl.BatchNormalization(epsilon=eps)(x)
    x = kl.ReLU()(x)
    x = kl.MaxPooling2D(pool_size=3, strides=2)(x)
    # 5 Part
    x = kl.Flatten()(x)
    x = kl.Dense(300, kernel_regularizer=l2(l2_reg_rate))(x)
    x = kl.BatchNormalization(epsilon=eps)(x)
    x = kl.ReLU()(x)
    x = kl.Dense(300)(x)
    x = kl.ReLU()(x)
    x = kl.Dense(class_count)(x)
    x = kl.Softmax()(x)
    model = Model(inputs=input_, outputs=x)

    opt = SGD(learning_rate=lr, momentum=0.9, nesterov=True)

    model.compile(optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy'])
    return model

def train(in_model, data, model_file='models/cnn2'):
    train_set, val_set, test_set, dataset_length = data
    if not os.path.exists(model_file):
        print("[INFO] Training Model ...")
        in_model.fit(train_set,
                    steps_per_epoch=dataset_length/BATCH_SIZE,
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