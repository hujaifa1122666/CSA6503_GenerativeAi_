# Install:
# pip install numpy pandas matplotlib scikit-learn torch transformers

import sys
import importlib

libraries = {
    "NumPy": "numpy",
    "Pandas": "pandas",
    "Matplotlib": "matplotlib",
    "Scikit-learn": "sklearn",
    "PyTorch": "torch",
    "Transformers": "transformers"
}

print("Python Version:", sys.version.split()[0])
print("\nInstalled AI/ML Library Versions")
print("---------------------------------")

for name, module_name in libraries.items():
    try:
        module = importlib.import_module(module_name)
        print(f"{name}: {module.__version__}")
    except ImportError:
        print(f"{name}: NOT INSTALLED")
