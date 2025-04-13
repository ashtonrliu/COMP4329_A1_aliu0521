import numpy as np

def standardize_data(data):
    # Calculate the mean and standard deviation for each feature
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    # Standardize the dataset
    standardized_data = (data - mean) / std
    return standardized_data

def normalize_data(data):
    # Calculate the minimum and maximum for each feature
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)

    # Normalize the dataset
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data
