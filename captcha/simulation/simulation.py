#!/bin/python3

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import random
import pickle
import tensorflow.keras as ker
from imageprocessor import process_set
from trainer import do_training
from solver import do_validation
from solver import rename_to_prediction

new_size = (20, 20)
model_file = "model.hdf5"
label_file = "labels.dat"
train_dirs = {
    "images": "../train_set",
    "output": "train_data/letters",
    "ignored": "train_data/ignored"
}
test_dirs = {
    "images": "../test_set",
    "output": "test_data/letters",
    "ignored": "test_data/ignored"
}

print()
print("Preprocess training and test sets")
# Preprocess train and test sets: save extracted letters, handle not recognised (ignored) images
process_set(train_dirs)
process_set(test_dirs)

labels = sorted(os.listdir(train_dirs["output"]))

# Save labels
with open(label_file, "wb") as file:
    pickle.dump(labels, file)

print()
print("Neural network training ...")
# Train neural network and save model parameters
do_training(train_dirs["output"], model_file, new_size, len(labels))

print()
print("Validation:")
# Fetch validating set and load model
validating_set = ker.preprocessing.image_dataset_from_directory(
    test_dirs["output"], image_size=new_size, label_mode="categorical", color_mode="grayscale", shuffle=True)
model = ker.models.load_model(model_file)

# Validate test set
model.evaluate(validating_set)

# Load labels and model
with open(label_file, "rb") as file:
    labels = pickle.load(file)
model = ker.models.load_model(model_file)

print()
print("Statistical validation:")
# Test set predictions (batch mode)
do_validation(test_dirs["images"], model, labels, (20, 20), batch_size=256)

print()
print("Captcha solving:")
# Random captcha predication (rename)
image_name = random.choice(os.listdir(test_dirs["images"]))
init_name = "unpredicted.png"
os.system("cp {} {}".format(os.path.join(os.path.relpath(test_dirs["images"]), image_name), init_name))
rename_to_prediction(os.path.join(os.path.relpath("."), init_name), model, labels, new_size)
result="Captcha {}{} renamed properly.".format(os.path.splitext(image_name)[0], "" if not os.path.isfile(init_name) and os.path.isfile(image_name) else " could not be")
print(result)
