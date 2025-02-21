# Factual Consistency Summer Project

** Describe the project here **

## Development

See instructions in DEVELOPMENT.md


## Install

To install this package first make sure you run `mwinit -o`. Then run the following commands:
```
git clone ssh://git.amazon.com/pkg/DomainAdaptationForFactualConsistencyDetection
cd DomainAdaptationForFactualConsistencyDetection
poetry install
```

# (optional) Install on MAC 

If you want to install locally on your MAC, you need to manually install torch. 
Following these steps to install torch:

Get the location of the virtual environment created by poetry. 
```
poetry env info
```

Suppose the virtual environment is located at `Library/Caches/pypoetry/domainadaptationforfactualconsistencydetec-ZrA-s4fn-py3.11`, then activate it using 
this command:

```
source ~/Library/Caches/pypoetry/virtualenvs/domainadaptationforfactualconsistencydetec-ZrA-s4fn-py3.11/bin/activate
```

Then install torch (following this instructions https://pytorch.org/get-started/locally/)
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -U sentence-transformers
```

# Test installation
To start jupyter notebook use this command: 

```
poetry run jupyter notebook 
```

# Run 
Run synthetic data algorithm with default parameters
```
poetry run python simple_genetic_algorithm.py --device='cpu'
```