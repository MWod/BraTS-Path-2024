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
import lightning as pl

from torchvision.models import ResNet18_Weights, ViT_B_16_Weights
### Internal Imports ###

from paths import pc_paths as p
from networks import resnet18, provgigapath, vit
from datasets import dataset as ds
from training import trainer
from experiments.pc_experiments import experiments as exp
from augmentation import augmentation as aug

########################



def exp_8(fold=1):
    dataset_path = p.raw_data_path / "Training"
    training_csv_path = p.csv_path / f"training_fold_{fold}.csv"
    validation_csv_path = p.csv_path / f"val_fold_{fold}.csv"

    training_augmentation_transforms = aug.transform_1()
    validation_augmentation_transforms = None
    training_network_transforms = provgigapath.transforms
    validation_network_transforms = provgigapath.transforms

    iteration_size = -1
    class_mapper = {'WM': 0, 'CT' : 1, 'PN': 2, 'NC': 3, 'MP': 4, 'IC': 5}
    num_classes = len(class_mapper)
    inverse_class_mapper = {0: 'WM', 1: 'CT', 2: 'PN', 3: 'NC', 4: 'MP', 5: 'IC'}

    training_dataset = ds.BraTSPathDataset(dataset_path, training_csv_path,
                                class_mapper, iteration_size, training_augmentation_transforms, training_network_transforms, mode='prov')
    validation_dataset = ds.BraTSPathDataset(dataset_path, validation_csv_path,
                                class_mapper, iteration_size, validation_augmentation_transforms, validation_network_transforms, mode='prov')
    
    print(f"Training Dataset size: {len(training_dataset)}")
    print(f"Validation Dataset size: {len(validation_dataset)}")
    print(f"Training Class counter: {training_dataset.class_counter}")
    print(f"Validation Class counter: {validation_dataset.class_counter}")
    print(f"Training Unique Weights: {np.unique(training_dataset.weights)}")

    batch_size = 32
    num_workers = 32
    sampler = tc.utils.data.WeightedRandomSampler(tc.from_numpy(training_dataset.weights).type('torch.DoubleTensor'), len(training_dataset.weights))
    training_dataloader = tc.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, sampler=sampler, num_workers=num_workers)
    validation_dataloader = tc.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    model = provgigapath.ProvGigaPath(num_classes=num_classes, checkpoint_path=p.provgigapath_path)
    
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

    checkpoint_name = {1: 'epoch=97_bacc.ckpt', 2: 'epoch=100_bacc.ckpt', 3: 'epoch=100_bacc.ckpt', 4: 'epoch=88_bacc.ckpt', 5: 'epoch=83_bacc.ckpt'}
    state_dict = load_state_dict(p.checkpoints_path / f"BraTS-Path_Exp7_Fold{fold}_Aug" / checkpoint_name[int(fold)])
    model.load_state_dict(state_dict)
    model.freeze_weights_all()
    model.unfreeze_weights(["fc"])
    print(f"Model: {model.model}")

    ### Parameters ###
    experiment_name = f"BraTS-Path_Exp8_Fold{fold}_Aug"
    learning_rate = 0.001
    save_step = 201
    to_load_checkpoint_path = None
    lr_decay = 0.995
    objective_function = tc.nn.CrossEntropyLoss()
    objective_function_params = {}
    optimizer_weight_decay = 0.005

    accelerator = 'gpu'
    devices = [1]
    num_nodes = 1
    logger = None
    callbacks = None
    max_epochs = 101
    accumulate_grad_batches = 1
    gradient_clip_val = 500
    reload_dataloaders_every_n_epochs = 100000
    precision = 'bf16-mixed'
    
    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['num_nodes'] = num_nodes
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['accumulate_grad_batches'] = accumulate_grad_batches
    lighting_params['gradient_clip_val'] = gradient_clip_val
    lighting_params['reload_dataloaders_every_n_epochs'] = reload_dataloaders_every_n_epochs
    lighting_params['precision'] = precision
    
    ### Parse Parameters ###
    training_params = dict()
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['model'] = model
    training_params['training_dataloader'] = training_dataloader
    training_params['validation_dataloader'] = validation_dataloader
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['lr_decay'] = lr_decay
    training_params['lightning_params'] = lighting_params
    training_params['num_classes'] = num_classes

    ### Cost functions and params
    training_params['objective_function'] = objective_function
    training_params['objective_function_params'] = objective_function_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay

    training_params['lightning_params'] = lighting_params

    ########################################
    return training_params