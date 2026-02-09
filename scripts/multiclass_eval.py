from hw02wildcatshawkeyes import deepl
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import getopt
import os
import glob


data_dir = "/home/pzs0001/hw02wildcatshawkeyes/data"
keyword = "hw02"

opts, args = getopt.getopt(sys.argv[1:], "", ["keyword=", "data_dir="])
for opt, val in opts:
    if opt == "--keyword":
        keyword = val
    elif opt == "--data_dir":
        data_dir = val

# Find all CSV files matching keyword
pattern = os.path.join(data_dir, f"{keyword}_metrics.csv")
files = glob.glob(pattern)

if not files:
    print(f"No files found matching {pattern}")
    sys.exit(1)

# Read and combine all matching CSVs
df = pd.concat([pd.read_csv(f) for f in files])

print(f"Found {len(df)} runs")

# Create boxplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

metrics = [
    ("train_accuracy", "test_accuracy", "Accuracy"),
    ("train_f1", "test_f1", "F1 Score"),
    ("train_precision", "test_precision", "Precision"),
    ("train_recall", "test_recall", "Recall"),
]

for ax, (train_col, test_col, title) in zip(axes.flatten(), metrics):
    data = [df[train_col], df[test_col]]
    ax.boxplot(data, labels=["Train", "Test"])
    ax.set_title(title)
    ax.set_ylabel(title)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = os.path.join(data_dir, f"{keyword}_boxplot_{timestamp}.png")
plt.savefig(filename)
print(f"Saved boxplot to {filename}")
plt.show()
