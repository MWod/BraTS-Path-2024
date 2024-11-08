### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Union, Callable
import collections

### External Imports ###
import numpy as np
import torch as tc
import timm

from torchsummary import summary
from torchvision import transforms

### Internal Imports ###
from paths import pc_paths as p

########################


transforms = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

class ProvGigaPath(tc.nn.Module):
    def __init__(self, num_classes, checkpoint_path):
        super().__init__()

        self.encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", checkpoint_path=checkpoint_path)
        self.fc = tc.nn.Sequential(
            tc.nn.Linear(1536, 256),
            tc.nn.PReLU(),
            tc.nn.Linear(256, num_classes)
        )
        self.model = tc.nn.Sequential(collections.OrderedDict(
            [
            ('encoder', self.encoder),
            ('fc', self.fc)
            ])
            )

    def forward(self, x):
        return self.model(x)
    
    def modify_weights(self, layer_names, value):
        for layer_name in layer_names:
            layer = getattr(self.model, layer_name)
            for param in layer.parameters():
                param.requires_grad = value

    def freeze_weights_all(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_weights(self, layer_names):
        self.modify_weights(layer_names, True)

    def unfreeze_weights_all(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def freeze_weights(self, layer_names):
        self.modify_weights(layer_names, False)

    def load_model(self, weights_path):
        self.model.load_state_dict(weights_path)
        self.model.eval()

    def load_model_from_checkpoint(self, checkpoint_path):
        checkpoint = tc.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def embedding_calculator(self):
        def calculator(x):
            modules = list(self.model.children())[:-1]
            model = tc.nn.Sequential(*modules)
            return model(x)
        return calculator

    def save_model(self, to_save_path):
        tc.save(self.model.state_dict(), to_save_path)

if __name__ == "__main__":
    model = ProvGigaPath(num_classes=6, checkpoint_path=p.provgigapath_path)
    print(model.model)
    summary(model.model.to("cuda:0"), (3, 224, 224))