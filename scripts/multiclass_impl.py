#!/usr/bin/env python3
"""
Script: multiclass_impl.py
Package: hw02wildcatshawkeyes
Course: CPE 487/587 - Machine Learning Tools
Homework: HW02

Usage:
    python scripts/multiclass_impl.py.py
"""

from hw02wildcatshawkeyes import example_function

def main():
    # Test data
    test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    # Run the function
    result = example_function(test_data)
    
    print(f"Input: {test_data}")
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
