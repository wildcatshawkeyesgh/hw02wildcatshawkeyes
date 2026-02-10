from hw02wildcatshawkeyes import deepl
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import getopt
import os


data_dir = "/home/pzs0001/hw02wildcatshawkeyes/data"
keyword = "hw02"

opts, args = getopt.getopt(sys.argv[1:], "", ["keyword=", "data_dir="])
for opt, val in opts:
    if opt == "--keyword":
        keyword = val
    elif opt == "--data_dir":
        data_dir = val


filename = os.path.join(data_dir, f"{keyword}_metrics.csv")
df = pd.read_csv(filename)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

plt.figure()
plt.boxplot([df["train_accuracy"], df["test_accuracy"]], labels=["Train", "Test"])
plt.title("Accuracy")
plt.savefig(os.path.join(data_dir, f"{keyword}_accuracy_boxplot_{timestamp}.png"))

plt.figure()
plt.boxplot([df["train_f1"], df["test_f1"]], labels=["Train", "Test"])
plt.title("F1 Score")
plt.savefig(os.path.join(data_dir, f"{keyword}_f1_boxplot_{timestamp}.png"))


plt.figure()
plt.boxplot([df["train_precision"], df["test_precision"]], labels=["Train", "Test"])
plt.title("Precision")
plt.savefig(os.path.join(data_dir, f"{keyword}_precision_boxplot_{timestamp}.png"))

plt.figure()
plt.boxplot([df["train_recall"], df["test_recall"]], labels=["Train", "Test"])
plt.title("Recall")
plt.savefig(os.path.join(data_dir, f"{keyword}_recall_boxplot_{timestamp}.png"))
