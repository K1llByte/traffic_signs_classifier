from PIL import Image
import numpy as np
from cnn1.cnn1 import fetch_data, IMAGE_SIZE, NUM_CLASSES, load_and_predict #, train , make_model 
from cnn2.cnn2 import make_model, train

# Fetch 'gtsrb' dataset and prepare data
data = fetch_data("data/gtsrb_full")

# # Make and compile Neural Network model
# model = make_model(NUM_CLASSES, IMAGE_SIZE)

# # Load a pretrained model if it exists
# model = train(model, data)


# pred = load_and_predict(model,"data/new_50_1.jpg")
# print(pred)