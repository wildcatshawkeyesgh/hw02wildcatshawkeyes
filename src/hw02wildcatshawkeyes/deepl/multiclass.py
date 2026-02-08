"""
Module: multiclass.py
Subpackage: deepl
Package: hw02wildcatshawkeyes
Course: CPE 487/587 - Machine Learning Tools
Homework: HW02

Description:
    Add your module description here.
"""

import torch
from torch import nn
import numpy as np
from typing import List, Tuple, Optional

__all__ = ["example_function"]


class SimpleNN(nn.Module):
    def __init__(self, in_features, m):  # m is the number of classifications
        super(SimpleNN, self).__init__()
        self.in_features = in_features
        self.fc1 = nn.Linear(self.in_features, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 16)
        self.fc4 = nn.Linear(16, m) # check m vs m-1
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ClassTrainer:
    def __init__(
        self, X_train, Y_train, eta, epoch,
        loss, optimizer, model, device):

        self.X_train = X_train
        self.Y_train = Y_train
        self.eta = eta
        self.epoch = epoch
        self.loss = loss
        self.optimizer = optimizer
        self.model = model
        self.device = device
        self.loss_vector = torch.zeros(epoch)
        self.accuracy_vector = torch.zeros(epoch)
    
    def train(self):
    
    def test(self, X_test, y_test):
    
    def predict(self, X):
    
    def save(self, file_name=None):
    
    def evaluation(self, loss_vector, accuracy_vector):
        
        

    
