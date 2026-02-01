# HW02 - wildcatshawkeyes

## Description

This package contains implementations for CPE 487/587 Machine Learning Tools Homework 02.

## Installation

### From PyPI
```bash
pip install hw02wildcatshawkeyes
```

### From GitHub Release
```bash
pip install https://github.com/wildcatshawkeyesgh/hw02wildcatshawkeyes/releases/download/v0.1.0/hw02wildcatshawkeyes-0.1.0-*.whl
```

### From Source
```bash
git clone https://github.com/wildcatshawkeyesgh/hw02wildcatshawkeyes
cd hw02wildcatshawkeyes
uv venv --python 3.12
source .venv/bin/activate
uv sync
uv build
pip install dist/*.whl
```

## Usage

```python
from hw02wildcatshawkeyes import example_function

result = example_function([1.0, 2.0, 3.0])
print(result)
```

## Package Structure

```
hw02wildcatshawkeyes/
├── src/
│   └── hw02wildcatshawkeyes/
│       ├── __init__.py
│       └── deepl/
│           ├── __init__.py
│           ├── two_layer_binary_classification.py.py
│           ├── multiclass.py.py
│       └── animation/
│           ├── __init__.py
│           ├── weight_animation.py.py
│           ├── largewt_animation.py.py
└── scripts/
    └── binaryclassification_animate_impl.py.py
    └── multiclass_impl.py.py
```

## Modules

### deepl
- `two_layer_binary_classification.py`: Add description here
- `multiclass.py`: Add description here

### animation
- `weight_animation.py`: Add description here
- `largewt_animation.py`: Add description here


## Scripts

- `binaryclassification_animate_impl.py.py`: Add description here
- `multiclass_impl.py.py`: Add description here

## Dependencies

- Python >= 3.12
- PyTorch
- NumPy
- Matplotlib

## Author

- **Keyword**: wildcatshawkeyes
- **Email**: wildcatshawkeyes@gmail.com
- **GitHub**: wildcatshawkeyesgh
- **Course**: CPE 487/587 Machine Learning Tools
- **Institution**: University of Alabama in Huntsville

## License

MIT License
