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
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint

### Internal Imports ###

from paths import pc_paths as p
from training import trainer as tr
from experiments.pc_experiments import experiments as exp

########################


def initialize(training_params):
    experiment_name = training_params['experiment_name']
    num_iterations = training_params['lightning_params']['max_epochs']
    save_step = training_params['save_step']
    checkpoints_path = os.path.join(p.checkpoints_path, experiment_name)
    best_loss_checkpoint = ModelCheckpoint(dirpath=checkpoints_path, filename='{epoch}_loss', save_top_k=1, mode='min', monitor='Loss/Validation/loss')
    best_f1_checkpoint = ModelCheckpoint(dirpath=checkpoints_path, filename='{epoch}_f1', save_top_k=1, mode='max', monitor='Loss/Validation/f1score')
    best_bacc_checkpoint = ModelCheckpoint(dirpath=checkpoints_path, filename='{epoch}_bacc', save_top_k=1, mode='max', monitor='Loss/Validation/bacc')
    best_precision_checkpoint = ModelCheckpoint(dirpath=checkpoints_path, filename='{epoch}_precision', save_top_k=1, mode='max', monitor='Loss/Validation/precision')
    best_recall_checkpoint = ModelCheckpoint(dirpath=checkpoints_path, filename='{epoch}_recall', save_top_k=1, mode='max', monitor='Loss/Validation/recall')
    best_auroc_checkpoint = ModelCheckpoint(dirpath=checkpoints_path, filename='{epoch}_auroc', save_top_k=1, mode='max', monitor='Loss/Validation/auroc')

    checkpoints_iters = list(range(0, num_iterations, save_step))
    checkpoints_iters = list(range(0, num_iterations, save_step))
    log_image_iters = list(range(0, num_iterations, save_step))
    if not os.path.isdir(checkpoints_path):
        os.makedirs(checkpoints_path)
    log_dir = os.path.join(p.logs_path, experiment_name)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    logger = pl_loggers.TensorBoardLogger(save_dir=log_dir, name=experiment_name)
    training_params['lightning_params']['logger'] = logger
    training_params['lightning_params']['callbacks'] = [best_loss_checkpoint, best_f1_checkpoint, best_bacc_checkpoint, best_precision_checkpoint, best_recall_checkpoint, best_auroc_checkpoint]  
    training_params['checkpoints_path'] = checkpoints_path
    training_params['checkpoint_iters'] = checkpoints_iters
    training_params['log_image_iters'] = log_image_iters
    return training_params

def run_training(training_params):
    training_params = initialize(training_params)
    trainer = tr.LightningTrainer(**training_params)
    trainer.run()

def run():
    for fold in range(1, 6):
        run_training(exp.exp_8(fold))
        
    pass


if __name__ == "__main__":
    run()