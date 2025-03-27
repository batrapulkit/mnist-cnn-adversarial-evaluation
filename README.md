# MNIST CNN Adversarial Evaluation

This project evaluates the performance of a Convolutional Neural Network (CNN) on the MNIST dataset, including its robustness against adversarial attacks. The goal is to demonstrate how adversarial examples affect the performance of a deep learning model, specifically a CNN trained on the MNIST handwritten digit dataset.

## Project Overview

The project contains the following key components:

- **CNN Model**: A simple Convolutional Neural Network trained on the MNIST dataset.
- **Adversarial Example Generation**: The code uses the Fast Gradient Sign Method (FGSM) to generate adversarial examples to test the model's robustness.
- **Evaluation**: The model is evaluated on both clean test data and adversarially perturbed data.
- **Visualization**: Original and adversarial images are displayed for visual comparison, and performance metrics such as accuracy are calculated and plotted.

## Requirements

- Python 3.6+
- TensorFlow 2.x
- Numpy
- Matplotlib

You can install the necessary dependencies using the following command:

```bash
pip install -r requirements.txt
```
```bash
python vuln.ipynb
```

## Key Functions
1. load_mnist_data(): Loads the MNIST dataset and normalizes the images to be between 0 and 1.

2. create_model(): Defines and compiles the CNN model for classification.

3. generate_adversarial_example(): Uses the Fast Gradient Sign Method (FGSM) to generate adversarial examples.

4. evaluate_model(): Evaluates the model on clean test data.

5. evaluate_on_adversarial_data(): Evaluates the model on adversarial test data.

6. plot_accuracy(): Plots the accuracy of the model on clean and adversarial data.

## Results
After training, the model's performance is evaluated on:

Clean Test Data: The accuracy of the model on the unperturbed MNIST test data.

Adversarial Test Data: The accuracy of the model on test data that has been perturbed using adversarial examples.
