# Technologies and Applications

## Overview

In this work we'll focus on developping a Convolutional neural network for the
*gtsrb* dataset (German Traffic Sign Dataset) to achieve better accuracy.

## Project Structure

Each neural network model we devoleped and tested will be packed in a module
to be reused by the classifier notebook (or other).

- `data/` - All image data including the target `gtsrb/` dataset and some images
for quick testing

- `models` - Saved Models to reuse

- `src/` - Contains all scripts and notebooks
    - `cnn1/` - First Neural Network module we used as a test


## Implementation

For our plans we have intentions of implementing various techniques:

- First, we will produce a variations on the dataset with some <ins>Data
Augmentation</ins> to improve the accuracy of images that have slight changes
in photo properties

- ...

___

The code will be organized as a script with functionalities grouped by:

<!-- - Imports -->
- Constant Parameters
- Dataset
    - Load Dataset
    - Prepare Dataset
    <!-- - Data Augmentation -->
- Model
    - Define Model
    - Train Model