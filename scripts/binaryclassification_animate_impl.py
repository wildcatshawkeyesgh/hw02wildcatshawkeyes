#!/usr/bin/env python3
"""
Script: binaryclassification_animate_impl.py
Package: hw02wildcatshawkeyes
Course: CPE 487/587 - Machine Learning Tools
Homework: HW02

Usage:
    python scripts/binaryclassification_animate_impl.py.py
"""


from hw02wildcatshawkeyes import deepl

losses, W1, W2, W3, W4 = deepl.binary_classification(200, 40000, epochs=50000)

print("Training complete.")
