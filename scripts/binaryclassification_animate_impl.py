#!/usr/bin/env python3
"""
Script: binaryclassification_animate_impl.py
Package: hw02wildcatshawkeyes
Course: CPE 487/587 - Machine Learning Tools
Homework: HW02

Usage:
    python scripts/binaryclassification_animate_impl.py.py
"""

import torch
from hw02wildcatshawkeyes import deepl
from hw02wildcatshawkeyes import animation

losses, W1, W2, W3, W4 = deepl.binary_classification(200, 40000, epochs=5000)


animation.animate_weight_heatmap(
    W1, dt=0.04, file_name="W1_animation", title_str="W1 Weight Heatmap"
)
animation.animate_weight_heatmap(
    W2, dt=0.04, file_name="W2_animation", title_str="W2 Weight Heatmap"
)
animation.animate_weight_heatmap(
    W3, dt=0.04, file_name="W3_animation", title_str="W3 Weight Heatmap"
)
animation.animate_weight_heatmap(
    W4, dt=0.04, file_name="W4_animation", title_str="W4 Weight Heatmap"
)
