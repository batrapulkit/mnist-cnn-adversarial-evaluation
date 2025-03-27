# src/data_poisoning.py

import numpy as np

def simulate_data_poisoning(x_train, y_train, poison_ratio=0.1, poison_class=1):
    """
    Simulate data poisoning by injecting poisoned data into the training set.
    
    Args:
    - x_train: The training data (images).
    - y_train: The corresponding training labels.
    - poison_ratio: Fraction of poisoned data to add.
    - poison_class: The class label of the poisoned data.
    
    Returns:
    - poisoned_x_train: The poisoned training data.
    - poisoned_y_train: The poisoned training labels.
    """
    num_poisoned = int(len(x_train) * poison_ratio)
    poisoned_images = np.random.normal(0, 1, size=(num_poisoned, 28, 28, 1))  # Random noise images
    poisoned_labels = np.full((num_poisoned,), poison_class)
    
    poisoned_x_train = np.concatenate([x_train, poisoned_images])
    poisoned_y_train = np.concatenate([y_train, poisoned_labels])
    
    return poisoned_x_train, poisoned_y_train
