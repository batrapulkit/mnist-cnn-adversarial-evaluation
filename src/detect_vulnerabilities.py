import sys
import os

# Add the 'src' directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))



import tensorflow as tf
from src.utils import generate_adversarial_example
from src.data_poisoning import simulate_data_poisoning
from src.model_inversion import simulate_model_inversion
import numpy as np

def detect_adversarial(model, x_test, y_test, epsilon=0.1):
    adversarial_accuracies = []
    for i in range(len(x_test)):
        image, label = x_test[i], y_test[i]
        adversarial_image, original_pred, adversarial_pred = generate_adversarial_example(model, image, label, epsilon)
        adversarial_accuracies.append(original_pred != adversarial_pred)
    return np.mean(adversarial_accuracies)

def detect_poisoning(model, x_train, y_train, x_test, y_test, poison_ratio=0.1):
    poisoned_x_train, poisoned_y_train = simulate_data_poisoning(x_train, y_train, poison_ratio)
    model.fit(poisoned_x_train, poisoned_y_train, epochs=5, batch_size=64, verbose=0)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    return test_acc

def detect_model_inversion(model, x_train, target_class=1, num_samples=100):
    inverted_images = simulate_model_inversion(model, x_train, target_class, num_samples)
    return inverted_images

def main():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Define model
    model = tf.keras.models.load_model('models/mnist_model.h5')

    # Adversarial detection
    adversarial_accuracy = detect_adversarial(model, x_test, y_test)
    print(f'Adversarial accuracy: {adversarial_accuracy}')

    # Data poisoning detection
    clean_accuracy = detect_poisoning(model, x_train, y_train, x_test, y_test)
    print(f'Accuracy after data poisoning: {clean_accuracy}')

    # Model inversion simulation
    inverted_images = detect_model_inversion(model, x_train)
    print(f'Inverted images: {inverted_images.shape}')

if __name__ == '__main__':
    main()
