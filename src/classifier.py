from PIL import Image
import numpy as np
from cnn1.cnn1 import fetch_data, IMAGE_SIZE, NUM_CLASSES, load_and_predict #, train , make_model 

def test_cnn1(epochs):
    from cnn1.cnn1 import make_model, train
    # Fetch 'gtsrb' dataset and prepare data
    data = fetch_data("data/gtsrb_full")

    # Make and compile Neural Network model
    model = make_model(NUM_CLASSES, IMAGE_SIZE)
    #model.summary()

    # Load a pretrained model if it exists
    model = train(model, data, model_file=f'models/cnn1_{epochs}epochs', num_epochs=epochs)

    return model

def test_cnn2(epochs):
    from cnn2.cnn2 import make_model, train
    # Fetch 'gtsrb' dataset and prepare data
    data = fetch_data("data/gtsrb_full")

    # Make and compile Neural Network model
    model = make_model(NUM_CLASSES, IMAGE_SIZE)
    #model.summary()

    # Load a pretrained model if it exists
    model = train(model, data, model_file=f'models/cnn2_{epochs}epochs', num_epochs=epochs)

    return model

def test_cnn3(epochs):
    from cnn3.cnn3 import make_model
    from cnn2.cnn2 import train
    # Fetch 'gtsrb' dataset and prepare data
    data = fetch_data("data/gtsrb_full")

    # Make and compile Neural Network model
    model = make_model(NUM_CLASSES, IMAGE_SIZE)
    #model.summary()

    # Load a pretrained model if it exists
    model = train(model, data, model_file=f'models/cnn3_{epochs}epochs', num_epochs=epochs)

    return model

#model = test_cnn1(epochs=20)
model = test_cnn2(epochs=72)
#model = test_cnn3(epochs=50)

pred = load_and_predict(model,"data/new_50_1.jpg")
print(pred)