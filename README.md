# Image Classifier Project
This project is an implementation of an image classifier built with PyTorch. The classifier can be trained on any set of labeled images. Here we focus on classifying different species of flowers. The model, once trained, can be used to predict the class of new images. There are 102 classes. I utilized 41 class in my notebook

## Setup
Ensure you have Python 3.8+ and PyTorch installed. You can install the required libraries using `pip`:
```sh
pip install torch torchvision
```

# Training the Classifier
To train the image classifier, use the `train.py` script. The script requires a data directory with three subdirectories: `train/`, `valid/`, and `test/`.
Here's an example command to start training:

```sh
python train.py "path/to/data" --save_dir "path/to/save/checkpoints" --arch "vgg16" --learning_rate 0.003 --hidden_units 1024 --output_size 102 --dropout 0.5 --epochs 20 --gpu
```

### Command-Line Arguments for train.py:
* `data_dir`: Path to the data directory.
* `--save_dir`: Path to the directory where checkpoints will be saved.
* `--arch`: Model architecture (default is "vgg16").
* `--learning_rate`: Learning rate for training.
* `--hidden_units`: Number of units in the hidden layer.
* `--output_size`: Number of classes in the output layer.
* `--dropout`: Dropout rate for training.
* `--epochs`: Number of epochs to train for.
* `--gpu`: Flag to enable GPU training (if available).

# Making Predictions
To predict the class of an image using a trained model checkpoint, use the predict.py script.
Here's an example command for making predictions:
```sh
python predict.py "path/to/image" "path/to/checkpoint" --top_k 3 --category_names "path/to/cat_to_name.json" --gpu
```

### Command-Line Arguments for predict.py:
* `input_image`: Path to the image file you want to predict.
* `checkpoint`: Path to the checkpoint file.
* `--top_k`: Return the top K most likely classes.
* `--category_names`: Path to a JSON file mapping categories to real names.
* `--gpu`: Flag to enable GPU inference (if available).

# Files in this Repository
* `train.py`: Script to train a new network.
* `predict.py`: Script to predict image class with a trained network.
* `train_utils.py`: Utility functions for training the network.
* `predict_utils.py`: Utility functions for making predictions.
* `cat_to_name.json`: JSON file mapping category labels to category names.

# Dataset
The dataset should be split into three parts: training, validation, and testing. Each part should have subdirectories for each category (i.e., type of flower). For example:
```sh
flowers_data/
  train/
    1/
    2/
    ...
  valid/
    1/
    2/
    ...
  test/
    1/
    2/
    ...
```
