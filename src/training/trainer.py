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

from monai import metrics
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torchmetrics.classification import F1Score, Accuracy, Precision, Recall, AUROC

### Internal Imports ###

from paths import pc_paths

########################

class LightningModule(pl.LightningModule):
    def __init__(self, training_params : dict, lightning_params : dict):
        super().__init__()
        ### General params
        self.model : tc.nn.Module = training_params['model']
        # self.logger = lightning_params['logger']
        self.learning_rate : float = training_params['learning_rate']
        self.optimizer_weight_decay : float = training_params['optimizer_weight_decay']
        self.lr_decay : float = training_params['lr_decay']
        
        ## Cost functions and params
        self.objective_function : Callable = training_params['objective_function']
        self.objective_function_params : dict = training_params['objective_function_params']

        ## Metrics
        self.f1_score_training = F1Score(task="multiclass", num_classes=training_params['num_classes'], average='macro')
        self.f1_score_validation = F1Score(task="multiclass", num_classes=training_params['num_classes'], average='macro')
        self.balanced_acc_training = Accuracy(task="multiclass", num_classes=training_params['num_classes'], average='macro')
        self.balanced_acc_validation = Accuracy(task="multiclass", num_classes=training_params['num_classes'], average='macro')
        self.precision_training = Precision(task="multiclass", num_classes=training_params['num_classes'], average='macro')
        self.precision_validation = Precision(task="multiclass", num_classes=training_params['num_classes'], average='macro')
        self.recall_training = Recall(task="multiclass", num_classes=training_params['num_classes'], average='macro')
        self.recall_validation = Recall(task="multiclass", num_classes=training_params['num_classes'], average='macro')
        self.auroc_training = AUROC(task="multiclass", num_classes=training_params['num_classes'], average='macro')
        self.auroc_validation = AUROC(task="multiclass", num_classes=training_params['num_classes'], average='macro')

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = tc.optim.AdamW(self.model.parameters(), self.learning_rate, weight_decay=self.optimizer_weight_decay)
        scheduler = tc.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: self.lr_decay ** epoch)
        dict = {'optimizer': optimizer, 'lr_scheduler': scheduler}
        return dict
    
    def training_step(self, batch, batch_idx):
        input_data, ground_truth = batch[0], batch[1]
        output = self.model(input_data)
        loss = self.objective_function(output, ground_truth, **self.objective_function_params)
        ### Logging ###
        f1 = self.f1_score_training(output, ground_truth)
        bacc = self.balanced_acc_training(output, ground_truth)
        prec = self.precision_training(output, ground_truth)
        rec = self.recall_training(output, ground_truth)
        auroc = self.auroc_training(output, ground_truth)
        self.log("Loss/Training/loss", loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Training/f1score", f1, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Training/bacc", bacc, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Training/precision", prec, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Training/recall", rec, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Training/auroc", auroc, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        return loss
                        
    def validation_step(self, batch, batch_idx):
        input_data, ground_truth = batch[0], batch[1]
        output = self.model(input_data)
        loss = self.objective_function(output, ground_truth, **self.objective_function_params)
        ### Logging ###
        f1 = self.f1_score_validation(output, ground_truth)
        bacc = self.balanced_acc_validation(output, ground_truth)
        prec = self.precision_validation(output, ground_truth)
        rec = self.recall_validation(output, ground_truth)
        auroc = self.auroc_validation(output, ground_truth)
        self.log("Loss/Validation/loss", loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/f1score", f1, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/bacc", bacc, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/precision", prec, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/recall", rec, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/auroc", auroc, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)

class LightningDataModule(pl.LightningDataModule):
    def __init__(self, training_dataloader, validation_dataloader):
        super().__init__()
        self.td = training_dataloader
        self.vd = validation_dataloader
        
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        self.td.dataset.shuffle()
        return self.td
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        self.vd.dataset.shuffle()
        return self.vd

class LightningTrainer():
    def __init__(self, **training_params : dict):
        ### General params
        self.training_dataloader : tc.utils.data.DataLoader = training_params['training_dataloader']
        self.validation_dataloader : tc.utils.data.DataLoader = training_params['validation_dataloader']
        lightning_params = training_params['lightning_params']    
        
        self.checkpoints_path : Union[str, pathlib.Path] = training_params['checkpoints_path']
        self.to_load_checkpoint_path : Union[str, pathlib.Path, None] = training_params['to_load_checkpoint_path']
        if self.to_load_checkpoint_path is None:
            self.module = LightningModule(training_params, lightning_params)
        else:
            self.load_checkpoint()
            
        self.trainer = pl.Trainer(**lightning_params)
        self.data_module = LightningDataModule(self.training_dataloader, self.validation_dataloader)

    def save_checkpoint(self) -> None:
        self.trainer.save_checkpoint(pathlib.Path(self.checkpoints_path) / "Last_Iteration")

    def load_checkpoint(self) -> None:
        self.module = LightningModule.load_from_checkpoint(self.to_load_checkpoint_path) 
    
    def run(self) -> None:
        self.trainer.fit(self.module, self.data_module)
        self.save_checkpoint()