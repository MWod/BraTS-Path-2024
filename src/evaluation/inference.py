### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pathlib
from typing import Union, Iterable, Any, Callable
import time
import random

### External Imports ###
import numpy as np
import torch as tc
import pandas as pd

from skimage import io
from PIL import Image


### Internal Imports ###

from paths import pc_paths as p

########################



def inference_single(image_path, model, mode, device, network_transforms=None):
    ### Load Image ###
    image = io.imread(image_path)
    if mode == 'direct':
        tensor = tc.from_numpy(image).permute(2, 0, 1)
        if network_transforms is not None:
            tensor = network_transforms(tensor)
    else:
        image_pil = Image.fromarray(image)
        if network_transforms is not None:
            tensor = network_transforms(image_pil)
            
    ### Run Inference ###
    output = model(tensor.unsqueeze(0).to(device))
    output = tc.softmax(output, dim=1)
    
    ### Return Output ###
    return output
    

def run():
    pass

if __name__ == '__main__':
    run()