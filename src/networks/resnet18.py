### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Union, Callable

### External Imports ###
import numpy as np
import torch as tc

from torchvision.models import resnet18, ResNet18_Weights
from torchsummary import summary

### Internal Imports ###

########################



class ResNet18(tc.nn.Module):
    def __init__(self, num_classes, weights=None):
        super().__init__()
        if weights is not None:
            self.model = resnet18(weights=weights)
        else:
            self.model = resnet18(weights=None)
        self.model.fc = tc.nn.Linear(512, num_classes)

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
    model = ResNet18(num_classes=2)
    print(model.model)
    summary(model.model.to("cuda:0"), (3, 224, 224))