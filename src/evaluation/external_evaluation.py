### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pathlib
from typing import Union, Iterable, Any, Callable
import time
import random
from collections import Counter

### External Imports ###
import numpy as np
import torch as tc
import pandas as pd

### Internal Imports ###

from paths import pc_paths as p

from networks import resnet18, provgigapath, vit
from augmentation import augmentation as aug
from evaluation import inference

from torchvision.models import ResNet18_Weights, ViT_B_16_Weights

########################



def run_inference(data_path, save_path, models, device, network_transforms=None):
    images = os.listdir(data_path)
    images = [item for item in images if ".png" in item or ".jpg" in item]
    # {'WM': 0, 'CT' : 1, 'PN': 2, 'NC': 3, 'MP': 4, 'IC': 5}
    #     One of: [0, 1, 2, 3, 4, 5] where:
    # - 0: CT
    # - 1: PN
    # - 2: MP
    # - 3: NC
    # - 4: IC
    # - 5: WM
    class_mapper = {1: 0, 2: 1, 4: 2, 3: 3, 5: 4, 0: 5}
    number_of_cases = len(images)
    outputs = []
    for idx in range(len(images)):
        print(f"Current case: {idx + 1} / {number_of_cases}")
        image_name = images[idx]
        image_path = data_path / image_name
        predictions = []
        with tc.no_grad():
            for model in models:
                output = inference.inference_single(image_path, model, mode='prov', device=device, network_transforms=network_transforms)
                prediction = tc.argmax(output).item()
                predictions.append(prediction)
        counter = Counter(predictions)
        print(f"Counter: {counter}")
        final_prediction = counter.most_common(1)[0][0] 
        print(f"Prediction: {final_prediction}")
        final_prediction = class_mapper[final_prediction]
        print(f"Prediction after transfer: {final_prediction}")
        to_append = (image_name, final_prediction)
        outputs.append(to_append)
    dataframe = pd.DataFrame(outputs, columns=['SubjectID', 'Prediction'])
    # dataframe.astype({'SubjectID' : 'str', 'Prediction' : 'int32'})
    dataframe.to_csv(save_path, index=False)



def run_inference_2(data_path, save_path, models, device, network_transforms=None):
    images = os.listdir(data_path)
    images = [item for item in images if ".png" in item or ".jpg" in item]
    # {'WM': 0, 'CT' : 1, 'PN': 2, 'NC': 3, 'MP': 4, 'IC': 5}
    #     One of: [0, 1, 2, 3, 4, 5] where:
    # - 0: CT
    # - 1: PN
    # - 2: MP
    # - 3: NC
    # - 4: IC
    # - 5: WM
    class_mapper = {1: 0, 2: 1, 4: 2, 3: 3, 5: 4, 0: 5}
    number_of_cases = len(images)
    outputs = []
    for idx in range(len(images)):
        print(f"Current case: {idx + 1} / {number_of_cases}")
        image_name = images[idx]
        image_path = data_path / image_name
        predictions = []
        with tc.no_grad():
            for idx, model in enumerate(models):
                if idx == 0:
                    output = inference.inference_single(image_path, model, mode='prov', device=device, network_transforms=network_transforms)
                else:
                    output += inference.inference_single(image_path, model, mode='prov', device=device, network_transforms=network_transforms)
                prediction = tc.argmax(output).item()
        print(f"Prediction: {prediction}")
        final_prediction = class_mapper[prediction]
        print(f"Prediction after transfer: {final_prediction}")
        to_append = (image_name, final_prediction)
        outputs.append(to_append)
    dataframe = pd.DataFrame(outputs, columns=['SubjectID', 'Prediction'])
    # dataframe.astype({'SubjectID' : 'str', 'Prediction' : 'int32'})
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
  

def load_resnet_state_dict(checkpoint_path):
    checkpoint = tc.load(checkpoint_path)
    state_dict  = checkpoint['state_dict']
    all_keys = list(state_dict.keys())
    output_state_dict = {}
    for key in all_keys:
        if "model" in key:
            output_state_dict[key.replace("model.model", "model")] = state_dict[key]
    return output_state_dict  

def load_provgigapath(checkpoint_path, device):
    num_classes = 6
    model = provgigapath.ProvGigaPath(num_classes=num_classes, checkpoint_path=p.provgigapath_path)
    state_dict = load_state_dict(checkpoint_path)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def load_resnet(checkpoint_path, device):
    num_classes = 6
    model = resnet18.ResNet18(num_classes=num_classes, weights=None)
    state_dict = load_resnet_state_dict(checkpoint_path)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

def load_vit(checkpoint_path, device):
    num_classes = 6
    model = vit.ViT16(num_classes=num_classes, weights=None)
    state_dict = load_resnet_state_dict(checkpoint_path)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def run():
    # device = "cuda:1"
    # network_transforms = provgigapath.transforms
    # data_path = p.raw_data_path / "Validation" / "Validation-Data-anoymized"
    # save_path = p.results_path / "Validation_Exp4_Results_DifferentInference.csv"
    # checkpoints_paths = []
    # checkpoints_paths.append(p.checkpoints_path / f"BraTS-Path_Exp4_Fold1_Aug" / 'epoch=85_bacc.ckpt')
    # checkpoints_paths.append(p.checkpoints_path / f"BraTS-Path_Exp4_Fold2_Aug" / 'epoch=100_bacc.ckpt')
    # checkpoints_paths.append(p.checkpoints_path / f"BraTS-Path_Exp4_Fold3_Aug" / 'epoch=99_bacc.ckpt')
    # checkpoints_paths.append(p.checkpoints_path / f"BraTS-Path_Exp4_Fold4_Aug" / 'epoch=96_bacc.ckpt')
    # checkpoints_paths.append(p.checkpoints_path / f"BraTS-Path_Exp4_Fold5_Aug" / 'epoch=76_bacc.ckpt')
    # models = []
    # for fold in range(0, 5):
    #     models.append(load_provgigapath(checkpoints_paths[fold], device))
    # run_inference_2(data_path, save_path, models, device, network_transforms=network_transforms)


    # device = "cuda:1"
    # network_transforms = ResNet18_Weights.IMAGENET1K_V1.transforms(antialias=True)
    # data_path = p.raw_data_path / "Validation" / "Validation-Data-anoymized"
    # save_path = p.results_path / "Validation_Exp5.csv"
    # checkpoints_paths = []
    # checkpoints_paths.append(p.checkpoints_path / f"BraTS-Path_Exp5_Fold1_Aug" / 'epoch=198_bacc.ckpt')
    # checkpoints_paths.append(p.checkpoints_path / f"BraTS-Path_Exp5_Fold2_Aug" / 'epoch=189_bacc.ckpt')
    # checkpoints_paths.append(p.checkpoints_path / f"BraTS-Path_Exp5_Fold3_Aug" / 'epoch=173_bacc.ckpt')
    # # checkpoints_paths.append(p.checkpoints_path / f"BraTS-Path_Exp4_Fold4_Aug" / 'epoch=96_bacc.ckpt')
    # # checkpoints_paths.append(p.checkpoints_path / f"BraTS-Path_Exp4_Fold5_Aug" / 'epoch=76_bacc.ckpt')
    # models = []
    # for fold in range(0, 3):
    #     # models.append(load_provgigapath(checkpoints_paths[fold], device))
    #     models.append(load_resnet(checkpoints_paths[fold], device))
    # run_inference_2(data_path, save_path, models, device, network_transforms=network_transforms)

    # device = "cuda:1"
    # network_transforms = ViT_B_16_Weights.IMAGENET1K_V1.transforms(antialias=True)
    # data_path = p.raw_data_path / "Validation" / "Validation-Data-anoymized"
    # save_path = p.results_path / "Validation_Exp6.csv"
    # checkpoints_paths = []
    # checkpoints_paths.append(p.checkpoints_path / f"BraTS-Path_Exp6_Fold1_Aug" / 'epoch=96_bacc.ckpt')
    # checkpoints_paths.append(p.checkpoints_path / f"BraTS-Path_Exp6_Fold2_Aug" / 'epoch=97_bacc.ckpt')
    # models = []
    # for fold in range(0, 2):
    #     # models.append(load_provgigapath(checkpoints_paths[fold], device))
    #     # models.append(load_resnet(checkpoints_paths[fold], device))
    #     models.append(load_vit(checkpoints_paths[fold], device))
    # run_inference_2(data_path, save_path, models, device, network_transforms=network_transforms)



    device = "cuda:1"
    network_transforms = provgigapath.transforms
    data_path = p.raw_data_path / "Validation" / "Validation-Data-anoymized"
    save_path = p.results_path / "Validation_Exp7_Results_DifferentInference.csv"
    checkpoints_paths = []
    checkpoints_paths.append(p.checkpoints_path / f"BraTS-Path_Exp7_Fold1_Aug" / 'epoch=97_bacc.ckpt')
    checkpoints_paths.append(p.checkpoints_path / f"BraTS-Path_Exp7_Fold2_Aug" / 'epoch=100_bacc.ckpt')
    checkpoints_paths.append(p.checkpoints_path / f"BraTS-Path_Exp7_Fold3_Aug" / 'epoch=100_bacc.ckpt')
    checkpoints_paths.append(p.checkpoints_path / f"BraTS-Path_Exp7_Fold4_Aug" / 'epoch=88_bacc.ckpt')
    checkpoints_paths.append(p.checkpoints_path / f"BraTS-Path_Exp7_Fold5_Aug" / 'epoch=83_bacc.ckpt')
    models = []
    for fold in range(0, 5):
        models.append(load_provgigapath(checkpoints_paths[fold], device))
    run_inference_2(data_path, save_path, models, device, network_transforms=network_transforms)



if __name__ == '__main__':
    run()