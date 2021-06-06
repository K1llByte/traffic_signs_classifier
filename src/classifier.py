from cnn1.cnn1 import fetch_data, IMAGE_SIZE, NUM_CLASSES #, train , make_model 
from utils import load_and_predict

def test_cnn1(epochs):
    from cnn1.cnn1 import make_model, train
    # Fetch 'gtsrb' dataset and prepare data
    data = fetch_data("data/gtsrb_full",data_augmentation=True)

    # Make and compile Neural Network model
    model = make_model(NUM_CLASSES, IMAGE_SIZE)
    #model.summary()

    # Train or load a pretrained model if it exists
    model = train(model, data, model_file=f'models/cnn1_{epochs}epochs', num_epochs=epochs)

    return model

def test_cnn2(epochs):
    from cnn2.cnn2 import make_model, train
    # Fetch 'gtsrb' dataset and prepare data
    data = fetch_data("data/gtsrb_full",data_augmentation=True)
    # from cnn1.cnn1 import data_augmentation
    # data = data_augmentation(data)

    # Make and compile Neural Network model
    model = make_model(NUM_CLASSES, IMAGE_SIZE)
    #model.summary()

    # Train or load a pretrained model if it exists
    model = train(model, data, model_file=f'models/cnn2_{epochs}epochs', num_epochs=epochs)

    return model

def test_cnn3(epochs):
    from cnn3.cnn3 import make_model
    from cnn2.cnn2 import train
    # from cnn1.cnn1 import data_augmentation

    # Fetch 'gtsrb' dataset and prepare data
    data = fetch_data("data/gtsrb_full",data_augmentation=True)
    # data = data_augmentation(data)


    # Make and compile Neural Network model
    model = make_model(NUM_CLASSES, IMAGE_SIZE)
    #model.summary()

    # Train or load a pretrained model if it exists
    model = train(model, data, model_file=f'models/cnn3_{epochs}epochs', num_epochs=epochs)

    return model

#model = test_cnn1(epochs=19)
#model = test_cnn2(epochs=10)
model = test_cnn3(epochs=100)

import os
to_predict = [ f'data/gold_tests/{f}' for f in os.listdir('data/gold_tests')]
pred = load_and_predict(model, to_predict)
print(pred)