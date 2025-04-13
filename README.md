# MLP Exploration

## Overview
This project implements a Multi-Layer Perceptron (MLP) along with a set of utility functions for data processing, cleaning, loading, and various helper functionalities. The codebase is organized into three main directories:
1. **utils/**
2. **model/**
3. **data/**

Below is an explanation of how each directory is structured and how they interact.

---

## 1. `utils/`
This directory contains standalone scripts with discrete functions used throughout the project. They include:

- **`data_cleaning.py`**  
  Contains functions for preprocessing and cleaning raw data. Used for data normalization and standardzsation

- **`data_loader.py`**  
  Provides utilities for loading and splitting data (training/test sets) from file. Preprocesses the output labels

- **`functions.py`**  
  A collection of helper functions that might be used across multiple files/notebooks. Contains mathematical functions, helper functions for data transformations, or any other shared utilities.

- **`notebook_display.py`**  
  Helper functions specifically tailored to enhancing display or visualization within Jupyter notebooks. Includes plots, and data metrics

---

## 2. `model/`
This directory contains the core neural network (MLP) logic. It is organized into the following main files:

- **`base.py`**  
  Houses the MLP class and its methods. This includes:
  - **Initialization**: Setting up network architecture, defining layers, and initializing weights/biases.  
  - **Forward Propagation**: Propagating input data through the network to generate predictions.  
  - **Backward Propagation**: Computing gradients and updating weights based on loss.  
  - **Testing/Evaluation**: Methods for evaluating model performance on validation/test data.

- **`layer.py`**  
  Defines a `Layer` class which is used to represent each layer within the MLP. This class handles layer-specific information such as:
  - Number of neurons
  - Activation function
  - Parameters for weights and biases
  - Any layer-specific computations used during forward or backward propagation

The `base.py` file typically instantiates and orchestrates multiple `Layer` objects from `layer.py` to form the entire MLP.

---

## 3. `data/`
This directory houses the input and output data used for training and testing the model. This includes:
- **Training Inputs and Outputs** (e.g. test_data.npy, train_data.npy)

---

## Installation
After cloning the repository, and creating a virtual environment, install the following dependencies

1. **Install Requirements**  
   Make sure you have all necessary libraries installed (e.g., numpy, pandas, etc.).  
   ```bash
   pip install -r requirements.txt
