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
from skimage import io
from PIL import Image

### Internal Imports ###

from helpers import utils as u
from augmentation import augmentation as aug

########################




class BraTSPathDataset(tc.utils.data.Dataset):
    """
    TODO
    """
    def __init__(
        self,
        data_path : Union[str, pathlib.Path],
        csv_path : Union[str, pathlib.Path],
        class_mapper : dict,
        iteration_size : int = -1,
        augmentation_transforms = None,
        network_transforms = None,
        mode='direct'):
        """
        TODO
        """
        self.data_path = data_path
        self.csv_path = csv_path
        self.iteration_size = iteration_size
        self.augmentation_transforms = augmentation_transforms
        self.network_transforms = network_transforms
        self.class_mapper = class_mapper
        self.mode = mode

        self.dataframe = pd.read_csv(csv_path)
        self.classes = list(self.class_mapper.keys())
        self.class_counter = self.calculate_samples()
        self.weights = self.get_weights()

    def __len__(self):
        if self.iteration_size < 0 or self.iteration_size >= len(self.dataframe):
            return len(self.dataframe)
        else:
            return self.iteration_size
    
    def calculate_samples(self):
        output_dict = {}
        for current_class in self.classes:
            output_dict[current_class] = np.sum(self.dataframe['Ground-Truth'] == current_class)
        return output_dict

    def get_weights(self):
        weights = [0] * len(self.dataframe)
        for idx in range(len(self.dataframe)):
            weights[idx] = 1.0 / self.class_counter[self.dataframe.iloc[idx]['Ground-Truth']]
        return np.array(weights, dtype=np.float64)
        
    def shuffle(self):
        if self.iteration_size > 0 or self.iteration_size >= len(self.dataframe):
            self.dataframe = self.dataframe.sample(n=len(self.dataframe), replace=False).reset_index(drop=True)
            self.weights = self.get_weights()

    def __getitem__(self, idx):
        current_case = self.dataframe.iloc[idx]
        image_path = self.data_path / current_case['Input Path']
        image_gt = current_case['Ground-Truth']
        gt = self.class_mapper[image_gt]
        image = io.imread(image_path)
        if self.augmentation_transforms is not None:
            image = aug.apply_transform(image, self.augmentation_transforms)
        if self.mode == 'direct':
            tensor = tc.from_numpy(image).permute(2, 0, 1)
            if self.network_transforms is not None:
                tensor = self.network_transforms(tensor)
        else:
            image_pil = Image.fromarray(image)
            if self.network_transforms is not None:
                tensor = self.network_transforms(image_pil)
        return tensor, gt
    