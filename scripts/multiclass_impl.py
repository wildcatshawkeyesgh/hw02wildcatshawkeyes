#!/usr/bin/env python3
"""
Script: multiclass_impl.py
Package: hw02wildcatshawkeyes
Course: CPE 487/587 - Machine Learning Tools
Homework: HW02
"""

from hw02wildcatshawkeyes import deepl
import pandas as pd
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.optim as optim
from datetime import datetime
import sys
import getopt
import os

file_location = "/home/pzs0001/hw02wildcatshawkeyes/data/Android_Malware.csv"
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(device)
eta = 0.01
epoch = 10000
keyword = "hw02"

opts, args = getopt.getopt(
    sys.argv[1:], "", ["file=", "eta=", "epoch=", "keyword=", "device="]
)
for opt, val in opts:
    if opt == "--file":
        file_location = val
    elif opt == "--eta":
        eta = float(val)
    elif opt == "--epoch":
        epoch = int(val)
    elif opt == "--keyword":
        keyword = val
    elif opt == "--device":
        device = val


columns_exclude = [
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Destination Port",
    "Protocol",
    "Timestamp",
    "Label",
]

all_columns = pd.read_csv(file_location, nrows=0).columns.tolist()
columns = [col for col in all_columns if col not in columns_exclude]

df_X = pd.read_csv(file_location, usecols=columns, low_memory=False)
df_y = pd.read_csv(file_location, usecols=["Label"])

print(df_X.isna().sum())
df_X = df_X.apply(pd.to_numeric, errors="coerce")

rows_before = len(df_X)
valid_rows = df_X.dropna().index


df_X = df_X.loc[valid_rows].reset_index(drop=True)
df_y = df_y.loc[valid_rows].reset_index(drop=True)

rows_dropped = rows_before - len(df_X)
print(f"Dropped {rows_dropped} rows with NaN values")
X_train, X_test, y_train, y_test = train_test_split(
    df_X, df_y, test_size=0.2, random_state=42
)

malware_classes = [
    "Android_Adware",
    "Android_Scareware",
    "Android_SMS_Malware",
    "Benign",
]


X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

label_encoder = LabelEncoder()
label_encoder.fit(malware_classes)
y_train_encoded = label_encoder.transform(y_train.values.ravel())
y_test_encoded = label_encoder.transform(y_test.values.ravel())
y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)


model = deepl.multiclass.SimpleNN(X_train_tensor.shape[1], len(malware_classes))
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=eta)
trainer = deepl.multiclass.ClassTrainer(
    X_train=X_train_tensor,
    Y_train=y_train_tensor,
    eta=eta,
    epoch=epoch,
    loss=loss,
    optimizer=optimizer,
    model=model,
    device=device,
)  # add params later
trainer.train()
train_acc, train_prec, train_rec, train_f1, test_acc, test_prec, test_rec, test_f1 = (
    trainer.evaluation(X_test_tensor, y_test_tensor)
)


df_metrics = pd.DataFrame(
    {
        "metric": ["accuracy", "precision", "recall", "f1"],
        "train": [train_acc, train_prec, train_rec, train_f1],
        "test": [test_acc, test_prec, test_rec, test_f1],
    }
)

data_dir = os.path.dirname(file_location)
filename = os.path.join(data_dir, f"{keyword}_metrics.csv")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

df_metrics = pd.DataFrame(
    {
        "timestamp": [timestamp],
        "eta": [eta],
        "epoch": [epoch],
        "train_accuracy": [train_acc],
        "train_precision": [train_prec],
        "train_recall": [train_rec],
        "train_f1": [train_f1],
        "test_accuracy": [test_acc],
        "test_precision": [test_prec],
        "test_recall": [test_rec],
        "test_f1": [test_f1],
    }
)
file_exists = os.path.exists(filename)
df_metrics.to_csv(filename, mode="a", header=not file_exists, index=False)
