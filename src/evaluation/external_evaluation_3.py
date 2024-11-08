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

### Internal Imports ###

from paths import pc_paths as p

from networks import resnet18, provgigapath, vit

from torchvision.models import ResNet18_Weights, ViT_B_16_Weights

########################



class BraTSPathDataset(tc.utils.data.Dataset):
    """
    TODO
    """
    def __init__(
        self,
        data_path,
        image_paths,
        network_transforms = None):
        """
        TODO
        """
        self.data_path = pathlib.Path(data_path)
        self.image_paths = image_paths
        self.network_transforms = network_transforms

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        current_case = self.image_paths[idx]
        image_path = current_case
        image = io.imread(self.data_path / image_path)
        image_pil = Image.fromarray(image)
        if self.network_transforms is not None:
            tensor = self.network_transforms(image_pil)
        return tensor, idx, image_path

def inference_batch(images, embedding, model, mode, device, network_transforms=None):
    ### Run Inference ###
    outputs = model(embedding)
    outputs = tc.softmax(outputs, dim=1)
    ### Return Output ###
    return outputs


def run_inference(data_path, save_path, encoder, models, device, network_transforms=None):
    images = os.listdir(data_path)
    images = [item for item in images if ".png" in item or ".jpg" in item]

    num_workers = 16
    batch_size = 64
    dataset = BraTSPathDataset(data_path, images, network_transforms)
    dataloader = tc.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=num_workers)
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
    all_outputs = [0] * number_of_cases

    for idx, (images, idxs, image_paths) in enumerate(dataloader):
        print(f"Current Batch: {idx + 1} / {number_of_cases // batch_size}")
        with tc.no_grad():
            embedding = encoder(images.to(device))
            for idx, model in enumerate(models):
                if idx == 0:
                    outputs = inference_batch(images, embedding, model, mode='prov', device=device, network_transforms=network_transforms)
                else:
                    outputs += inference_batch(images, embedding, model, mode='prov', device=device, network_transforms=network_transforms)
                prediction = tc.argmax(outputs, dim=1)
        for j in range(len(idxs)):
            current_idx = idxs[j]
            pred = prediction[j].item()
            image_name = image_paths[j]
            print(f"Prediction: {pred}")
            final_prediction = class_mapper[pred]
            print(f"Prediction after transfer: {final_prediction}")
            to_append = (image_name, final_prediction)
            all_outputs[current_idx] = to_append
    dataframe = pd.DataFrame(all_outputs, columns=['SubjectID', 'Prediction'])
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


def run():
    device = "cuda:1"
    network_transforms = provgigapath.transforms
    data_path = p.raw_data_path / "Validation" / "Validation-Data-anoymized"
    save_path = p.results_path / "Validation_Exp7_Results_DifferentInference_V3.csv"
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