### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pathlib
from typing import Union, Callable
import time
import random

### External Imports ###
import numpy as np
import torch as tc
import pandas as pd
from sklearn.model_selection import KFold

### Internal Imports ###

from paths import pc_paths as p

########################

def parse_dataset():
    data_path = p.raw_data_path / "Training"
    output_csv_path = p.csv_path / "dataset.csv"

    classes = os.listdir(data_path)
    dataframe = []
    for current_class in classes:
        current_images = os.listdir(data_path / current_class)
        current_images = [item for item in current_images if ".jpg" in item or ".png" in item]
        print(f"Number of images in class {current_class}: {len(current_images)}")
        for current_image in current_images:
            input_path = pathlib.Path(current_class) / current_image
            ground_truth = current_class
            to_append = [input_path, ground_truth]
            dataframe.append(to_append)
    dataframe = pd.DataFrame(dataframe, columns=['Input Path', 'Ground-Truth'])
    dataframe.to_csv(output_csv_path, index=False)


def split_dataframe(num_folds=5, seed=1234):
    input_csv_path = p.csv_path / "dataset.csv"
    output_splits_path = p.csv_path
    if not os.path.isdir(os.path.dirname(output_splits_path)):
        os.makedirs(os.path.dirname(output_splits_path))
    dataframe = pd.read_csv(input_csv_path)
    print(f"Dataset size: {len(dataframe)}")
    kf = KFold(n_splits=num_folds, shuffle=True)
    folds = kf.split(dataframe)
    for fold in range(num_folds):
        train_index, test_index = next(folds)
        current_training_dataframe = dataframe.loc[train_index]
        current_validation_dataframe = dataframe.loc[test_index]
        print(f"Fold {fold + 1} Training dataset size: {len(current_training_dataframe)}")
        print(f"Fold {fold + 1} Validation dataset size: {len(current_validation_dataframe)}")
        training_csv_path = output_splits_path / f"training_fold_{fold+1}.csv"
        validation_csv_path = output_splits_path / f"val_fold_{fold+1}.csv"
        current_training_dataframe.to_csv(training_csv_path, index=False)
        current_validation_dataframe.to_csv(validation_csv_path, index=False)

def run():
    # parse_dataset()
    # split_dataframe()

    pass

if __name__ == "__main__":
    run()
