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

## Results

| Net       | Epochs | Accuracy | Predicts 50 |
|:---------:|:------:|:--------:|:-----------:|
| cnn1      | 20     | 97.32%   | yes         |
| cnn2      | 50     | 93.42%   | no          |
| cnn2      | 72     | 95.36%   | no          |
| cnn2 (DA) | 150    | 94.89%   | no          |
| cnn3      | 20     | 95.49%   | no          |
| cnn3      | 50     | 95.29%   | no          |


<!--
- Introdução
    - Objectivo
    - Resumo das aproches
- Estrutura do projeto
    - (de modo a fornecer modularidade e reutilizar componentes desenvolvidas bla bla bla)
- Redes
    - Convolutional Neural Network 1
        - (Dizer que foi a nossa primeira aproach e que experimentamos com variações)
    - Convolutional Neural Network 2
        - (Mencionar que foi um bom recurso para entender de que forma se pode desenvolver redes para este dataset/tópico)
    - Convolutional Neural Network 3
        - (Dizer que nos sentimos inspirados para desenvolver uma rede nossa que fosse trazer resultados mais fiaveis)
    - Results
        - Nosso caso de ouro (limit 50)
        - (Notar que afinal somos uns falhados e que a rede do stor não só dá melhor accuracy como também consegue identificar corretamente o nosso caso de ouro)
- Conclusão
-->