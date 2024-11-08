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

from torchsummary import summary
from torchvision.models import vision_transformer, ViT_B_16_Weights

### Internal Imports ###

########################


class ViT16(tc.nn.Module):
    def __init__(self, num_classes, weights=None):
        super().__init__()
        if weights is not None:
            self.model = vision_transformer.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            self.model = vision_transformer.vit_b_16(weights=None)
        self.embedding_size = 768

        heads_layers: collections.OrderedDict[str, tc.nn.Module] = collections.OrderedDict()
        heads_layers["pre_logits"] = tc.nn.Linear(self.embedding_size, self.embedding_size)
        heads_layers["act"] = tc.nn.Tanh()
        heads_layers["head"] = tc.nn.Linear(self.embedding_size, num_classes)
        heads = tc.nn.Sequential(heads_layers)
        self.model.heads = heads
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.model(x)

    def modify_weights(self, layer_names, value):
        for layer_name in layer_names:
            layer = getattr(self.model, layer_name)
            for param in layer.parameters():
                param.requires_grad = value

    def modify_encoder_weights(self, layer_names, value):
        for layer_name in layer_names:
            layer = getattr(getattr(getattr(self.model, 'encoder'), 'layers'), layer_name)
            for param in layer.parameters():
                param.requires_grad = value   
    
    def modify_normalization_weights(self, value):
        norm_layer = getattr(getattr(self.model, "encoder"), "ln")
        for param in norm_layer.parameters():
            param.requires_grad = True

    def unfreeze_weights_all(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def unfreeze_weights(self, layer_names):
        self.modify_weights(layer_names, True)
        self.modify_normalization_weights(True)

    def freeze_weights(self, layer_names):
        self.modify_weights(layer_names, False)
        self.modify_normalization_weights(False)

    def unfreeze_encoder_weights(self, layer_names):
        self.modify_encoder_weights(layer_names, True)

    def freeze_encoder_weights(self, layer_names):
        self.modify_encoder_weights(layer_names, False)

    def load_model(self, weights_path):
        self.model.load_state_dict(weights_path)
        self.model.eval()

    def load_model_from_checkpoint(self, checkpoint_path):
        checkpoint = tc.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def embedding_calculator(self):
        def get_activation(activation, name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        def calculator(x):
            activation = {}
            getattr(self.model.heads, 'pre_logits').register_forward_hook(get_activation(activation, 'pre_logits'))
            x = self.model(x)
            output = activation['pre_logits']
            return output.unsqueeze(-1).unsqueeze(-1)
        return calculator

    def save_model(self, to_save_path):
        tc.save(self.model.state_dict(), to_save_path)

if __name__ == "__main__":
    model = ViT16(2)
    print(model.model)