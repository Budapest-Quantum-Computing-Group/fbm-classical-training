# Classical training of Fermionic Born Machines


# Installation

To install all dependencies, execute
```bash
pip install -r requirements.txt
```

# Test data

For the paper, four test datasets were used, found under `data/`. Here, each subdirectory contains a `test_set.txt` and `training_set.txt` file by default, except for `data/mrf/` as it contains further subdirectories `0/`-`9/`, each with `test_set.txt` and `training_set.txt` files.

# Trainings

## Small-scale FBMs
To train the FBM model using all Z-string expectation values up to a fixed locality cutoff, computing the total variation distance in each training step, run:
```bash
python scripts/exact_fbm_training.py
```
## Training FBMs
To train the FBM model using a fixed number of randomly sampled Z-strings in each step and finally test the model on the test set, run:
```bash
python scripts/fbm_training.py
```

## Training free-FBMs
To train the free-FBM model using a fixed number of randomly sampled Z-strings in each step and finally test the model on the test set, run:
```bash
python scripts/free_fbm_training.py
```

## Fitting and training reference models for benchmarks
To obtain benchmark data from the nonparametric models (random samples, statistical Chow-Liu approximation, discrepancy between training and test sets) run:
```bash
python benchmarks/nonparametric_models.py
```
To tune, and train the Restricted Boltzmann Machine, run
```bash
python benchmarks/train_rbm.py
```
## Changing hyperparameters
To change hyperparameters and problem path, modify the `PATH` and `CONFIG` variables in the corresponding python files.