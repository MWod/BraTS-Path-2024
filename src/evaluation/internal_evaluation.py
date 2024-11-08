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

### Internal Imports ###

from paths import pc_paths as p

from networks import resnet18, provgigapath
from augmentation import augmentation as aug
from evaluation import inference

########################


def run_inference(data_path, save_path, model, fold, device, network_transforms=None):
    dataframe_path = p.csv_path / f"val_fold_{fold}.csv"
    dataframe = pd.read_csv(dataframe_path)
    number_of_cases = len(dataframe)
    class_mapper = {'WM': 0, 'CT' : 1, 'PN': 2, 'NC': 3, 'MP': 4, 'IC': 5}
    outputs = []
    for idx in range(number_of_cases):
        print(f"Current case: {idx + 1} / {number_of_cases}")
        current_case = dataframe.iloc[idx]        
        image_path = data_path / current_case['Input Path']
        gt = class_mapper[current_case['Ground-Truth']]
        with tc.no_grad():
            output = inference.inference_single(image_path, model, mode='prov', device=device, network_transforms=network_transforms)
        prediction = tc.argmax(output).item()
        class_0 = output[0, 0].item()
        class_1 = output[0, 1].item()
        class_2 = output[0, 2].item()
        class_3 = output[0, 3].item()
        class_4 = output[0, 4].item()
        class_5 = output[0, 5].item()
        to_append = (current_case['Input Path'], gt, prediction, class_0, class_1, class_2, class_3, class_4, class_5)
        outputs.append(to_append)
        
    dataframe = pd.DataFrame(outputs, columns=['Input Path', 'Ground-Truth', 'Prediction', 'Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5'])
    dataframe.to_csv(save_path, index=False)
    
def load_state_dict(checkpoint_path):
    checkpoint = tc.load(checkpoint_path)
    state_dict  = checkpoint['state_dict']
    all_keys = list(state_dict.keys())
    output_state_dict = {}
    for key in all_keys:
        if "model.m" in key:
            output_state_dict[key.replace("model.m", "m")] = state_dict[key]
        if "encoder" in key:
            output_state_dict[key.replace("model.", "")] = state_dict[key]
        if "fc" in key:
            output_state_dict[key.replace("model.", "")] = state_dict[key]
    return output_state_dict  
  
def load_provgigapath(checkpoint_path, device):
    num_classes = 6
    model = provgigapath.ProvGigaPath(num_classes=num_classes, checkpoint_path=p.provgigapath_path)
    state_dict = load_state_dict(checkpoint_path)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model
    
def run():
    device = "cuda:0"
    data_path = p.raw_data_path / "Training"
    
    
    fold = 1
    checkpoint_path = p.checkpoints_path / f"BraTS-Path_Exp4_Fold{fold}_Aug" / 'epoch=85_bacc.ckpt'
    save_path = p.results_path / f"ProvGigaPath_Exp4_Fold{fold}.csv"
    network_transforms = provgigapath.transforms
    model = load_provgigapath(checkpoint_path, device)
    run_inference(data_path, save_path, model, fold, device, network_transforms)

    fold = 2
    checkpoint_path = p.checkpoints_path / f"BraTS-Path_Exp4_Fold{fold}_Aug" / 'epoch=100_bacc.ckpt'
    save_path = p.results_path / f"ProvGigaPath_Exp4_Fold{fold}.csv"
    network_transforms = provgigapath.transforms
    model = load_provgigapath(checkpoint_path, device)
    run_inference(data_path, save_path, model, fold, device, network_transforms)
    
    fold = 3
    checkpoint_path = p.checkpoints_path / f"BraTS-Path_Exp4_Fold{fold}_Aug" / 'epoch=99_bacc.ckpt'
    save_path = p.results_path / f"ProvGigaPath_Exp4_Fold{fold}.csv"
    network_transforms = provgigapath.transforms
    model = load_provgigapath(checkpoint_path, device)
    run_inference(data_path, save_path, model, fold, device, network_transforms)
    
    fold = 4
    checkpoint_path = p.checkpoints_path / f"BraTS-Path_Exp4_Fold{fold}_Aug" / 'epoch=96_bacc.ckpt'
    save_path = p.results_path / f"ProvGigaPath_Exp4_Fold{fold}.csv"
    network_transforms = provgigapath.transforms
    model = load_provgigapath(checkpoint_path, device)
    run_inference(data_path, save_path, model, fold, device, network_transforms)
    
    fold = 5
    checkpoint_path = p.checkpoints_path / f"BraTS-Path_Exp4_Fold{fold}_Aug" / 'epoch=76_bacc.ckpt'
    save_path = p.results_path / f"ProvGigaPath_Exp4_Fold{fold}.csv"
    network_transforms = provgigapath.transforms
    model = load_provgigapath(checkpoint_path, device)
    run_inference(data_path, save_path, model, fold, device, network_transforms)
    
if __name__ == '__main__':
    run()