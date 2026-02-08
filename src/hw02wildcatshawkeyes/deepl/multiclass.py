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
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    ConfusionMatrixDisplay,
)

__all__ = ["SimpleNN", "ClassTrainer"]


class SimpleNN(nn.Module):
    def __init__(self, in_features, m):  # m is the number of classifications
        super(SimpleNN, self).__init__()
        self.in_features = in_features
        self.fc1 = nn.Linear(self.in_features, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 16)
        self.fc4 = nn.Linear(16, m)  # check m vs m-1
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ClassTrainer:
    def __init__(self, X_train, Y_train, eta, epoch, loss, optimizer, model, device):

        self.X_train = X_train
        self.Y_train = Y_train
        self.eta = eta
        self.epoch = epoch
        self.loss = loss
        self.optimizer = optimizer
        self.model = model
        self.device = device
        self.loss_vector = torch.none(epoch)
        self.accuracy_vector = torch.none(epoch)

    def train(self):
        for i in range(self.epoch):
            self.optimizer.zero_grad()
            predictions = self.model.forward(self.X_train)
            loss = self.loss(predictions, self.Y_train)
            loss.backward()
            self.optimizer.step()

            self.loss_vector[i] = loss.item()

            predicted_classes = torch.argmax(predictions, dim=1)
            correct = (predicted_classes == self.Y_train).float()

            self.accuracy_vector[i] = torch.mean(correct).item()

    def test(self, X_test, y_test):
        with torch.no_grad():
            test_outputs = self.model(X_test)
            test_loss = self.loss(test_outputs, y_test)

    def predict(self, X):
        with torch.no_grad():
            y_pred = self.model(X)
            predicted_classes = torch.argmax(y_pred, dim=1)
            return predicted_classes

    def save(self, file_name=None):
        if file_name is None:
            file_name = "model.onnx"
        dummy_input = torch.zeros(1, self.X_train.shape[1])
        torch.onnx.export(self.model, dummy_input, file_name)

    def evaluation(self, loss_vector, accuracy_vector):

        plt.figure()
        plt.plot(self.loss_vector.numpy())
        plt.title("Loss over epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig("loss_curve.png")
        plt.show()

        # Plot accuracy
        plt.figure()
        plt.plot(self.accuracy_vector.numpy())
        plt.title("Accuracy over epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.savefig("accuracy_curve.png")
        plt.show()

        # Get predictions
        train_preds = self.predict(self.X_train)
        test_preds = self.predict(X_test)

        # Confusion matrix
        plt.figure()
        cm = confusion_matrix(y_test.numpy(), test_preds.numpy())
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        plt.savefig("confusion_matrix.png")
        plt.show()

        # Training metrics
        train_accuracy = accuracy_score(self.Y_train.numpy(), train_preds.numpy())
        train_precision = precision_score(
            self.Y_train.numpy(), train_preds.numpy(), average="weighted"
        )
        train_recall = recall_score(
            self.Y_train.numpy(), train_preds.numpy(), average="weighted"
        )
        train_f1 = f1_score(
            self.Y_train.numpy(), train_preds.numpy(), average="weighted"
        )

        # Test metrics
        test_accuracy = accuracy_score(y_test.numpy(), test_preds.numpy())
        test_precision = precision_score(
            y_test.numpy(), test_preds.numpy(), average="weighted"
        )
        test_recall = recall_score(
            y_test.numpy(), test_preds.numpy(), average="weighted"
        )
        test_f1 = f1_score(y_test.numpy(), test_preds.numpy(), average="weighted")

        return (
            train_accuracy,
            train_precision,
            train_recall,
            train_f1,
            test_accuracy,
            test_precision,
            test_recall,
            test_f1,
        )
