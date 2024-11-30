## Project Overview
This project implements a Physics-Informed Neural Network (PINN) framework for solving differential equations with various boundary conditions. The framework includes components for data generation, model training, evaluation, and exporting models to ONNX format. The project is structured to handle different boundary conditions, including Dirichlet-Dirichlet (DD), Dirichlet-Neumann (DN), Neumann-Dirichlet (ND), and Neumann-Neumann (NN).

## Key Components

### Data Generation

The [`DataGenerator`](data.py) class in [`data.py`](data.py) is responsible for generating and saving input and output data for training the models. It includes methods for generating right-hand side values, boundary conditions, coefficients, and solutions.

### Model Definition

The [`Network`](model.py) and [`Network_without_coeff`](model.py) classes in [`model.py`](model.py) define the neural network architectures used in the PINN framework. These classes include methods for forward propagation and printing the parameter count.

### Training

The [`Trainer`](compare.py) class in [`trainer.py`](trainer.py) handles the training process of the PINN models. It includes methods for loading data, training epochs, validation, and saving checkpoints.

### Evaluation

The [`Evaluator`](compare.py) class in [`evaluator.py`](evaluator.py) is used to evaluate the trained models. It includes methods for loading models, formatting data, moving data to the device, and plotting results.

### Exporting Models

The [`Exporter`](onnx.py) class in [`onnx.py`](onnx.py) provides functionality to export trained models to ONNX format. It includes methods for loading models, moving data to the device, and exporting models.

### Optimization

The [`optimize.py`](optimize.py) script uses Optuna to perform hyperparameter optimization for the PINN models. It includes functions for parsing validation loss, modifying configuration files, and deleting remaining checkpoints.

## Running the Project

### Training
To train a model with specific boundary conditions, run the corresponding script in the run directory. For example, to train a model with Dirichlet-Dirichlet boundary conditions, run:
```
python run/DD.py
```

Evaluation
To evaluate a trained model, run the `evaluator.py` script:
```
python evaluator.py
```

Exporting Models
To export a trained model to ONNX format, run the `onnx.py` script:
```
python onnx.py
```

Optimization
To perform hyperparameter optimization, run the `optimize.py` script:
```
python optimize.py --epochs 15 --bc_type DN --objective architecture
```

Note
The Neumann-Neumann framework referenced in qg_neumann_neumann_pinn.py is not part of this repository. This file includes code related to components that are not included in the current repository.