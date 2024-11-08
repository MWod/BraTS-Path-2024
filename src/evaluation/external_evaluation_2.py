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
import collections

### External Imports ###
import numpy as np
import torch as tc
import pandas as pd
from skimage import io
from PIL import Image
from torchvision import transforms
### Internal Imports ###

from paths import pc_paths as p

########################

prov_transforms = transforms.Compose(
[
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def inference_single(embedding, model, mode, device, network_transforms=None):    
    ### Run Inference ###
    output = model(embedding)
    output = tc.softmax(output, dim=1)
    ### Return Output ###
    return output


def run_inference(data_path, save_path, encoder, models, device, network_transforms=None):
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
        image = io.imread(image_path)
        image_pil = Image.fromarray(image)
        tensor = network_transforms(image_pil)
        with tc.no_grad():
            embedding = encoder(tensor.unsqueeze(0).to(device))
            for idx, model in enumerate(models):
                if idx == 0:
                    output = inference_single(embedding, model, mode='prov', device=device, network_transforms=network_transforms)
                else:
                    output += inference_single(embedding, model, mode='prov', device=device, network_transforms=network_transforms)
                prediction = tc.argmax(output).item()
        print(f"Prediction: {prediction}")
        final_prediction = class_mapper[prediction]
        print(f"Prediction after transfer: {final_prediction}")
        to_append = (image_name, final_prediction)
        outputs.append(to_append)
    dataframe = pd.DataFrame(outputs, columns=['SubjectID', 'Prediction'])
    dataframe.to_csv(save_path, index=False)

def load_encoder(device):
    model = tc.load(p.checkpoints_path / "ProvGigaPath")
    model = model.to(device)
    model.eval()
    return model


class Classifier(tc.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc = tc.nn.Sequential(
            tc.nn.Linear(1536, 256),
            tc.nn.PReLU(),
            tc.nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.fc(x)
    
def load_classifier(checkpoint_path, device):
    num_classes = 6
    model = Classifier(num_classes)
    state_dict = tc.load(checkpoint_path)
    output_state_dict = {}
    for key in state_dict.keys():
        if "model" in key:
            output_state_dict[key.replace("model.", "")] = state_dict[key]
    model.load_state_dict(output_state_dict)
    model = model.to(device)
    model.eval()
    return model


def parse_checkpoint(input_checkpoint_path, output_checkpoint_path):
    checkpoint = tc.load(input_checkpoint_path)
    state_dict  = checkpoint['state_dict']
    all_keys = list(state_dict.keys())
    output_state_dict = {}
    for key in all_keys:
        if "fc" in key and "encoder" not in key:
            output_state_dict[key.replace("model.model", "model")] = state_dict[key]
    print(output_state_dict.keys())
    tc.save(output_state_dict, output_checkpoint_path)

def parse_encoder(device):
    encoder = load_encoder(device)
    tc.save(encoder, p.checkpoints_path / "ProvGigaPath")


def run():
    device = "cuda:0"
    network_transforms = prov_transforms
    data_path = p.raw_data_path / "Validation" / "Validation-Data-anoymized"
    save_path = p.results_path / "Validation_Exp7_Results_DifferentInference_V4.csv"
    encoder = load_encoder(device)
    checkpoints_paths = []
    checkpoints_paths.append(p.checkpoints_path / f"BraTS-Path_Exp7_Fold1")
    checkpoints_paths.append(p.checkpoints_path / f"BraTS-Path_Exp7_Fold2")
    checkpoints_paths.append(p.checkpoints_path / f"BraTS-Path_Exp7_Fold3")
    checkpoints_paths.append(p.checkpoints_path / f"BraTS-Path_Exp7_Fold4")
    checkpoints_paths.append(p.checkpoints_path / f"BraTS-Path_Exp7_Fold5")
    models = []
    for fold in range(0, 5):
        models.append(load_classifier(checkpoints_paths[fold], device))
    run_inference(data_path, save_path, encoder, models, device, network_transforms=network_transforms)

    pass



if __name__ == '__main__':
    run()