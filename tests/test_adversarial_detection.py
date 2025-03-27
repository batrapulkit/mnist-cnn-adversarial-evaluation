# tests/test_adversarial_detection.py

import unittest
import numpy as np
import tensorflow as tf
from src.utils import generate_adversarial_example

class TestAdversarialDetection(unittest.TestCase):
    
    def setUp(self):
        # Load MNIST model for testing
        self.model = tf.keras.models.load_model('models/mnist_model.h5')
        (_, _), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()
        self.x_test = self.x_test / 255.0
    
    def test_adversarial_accuracy(self):
        epsilon = 0.1
        adversarial_accuracy = 0
        for i in range(100):  # Testing on a subset
            image = self.x_test[i]
            label = self.y_test[i]
            _, original_pred, adversarial_pred = generate_adversarial_example(self.model, image, label, epsilon)
            adversarial_accuracy += (original_pred != adversarial_pred)
        adversarial_accuracy = adversarial_accuracy / 100
        self.assertLess(adversarial_accuracy, 0.5, "Adversarial accuracy should be below 50%")

if __name__ == '__main__':
    unittest.main()
