# tests/test_model_inversion.py

import unittest
import tensorflow as tf
from src.model_inversion import simulate_model_inversion

class TestModelInversion(unittest.TestCase):

    def setUp(self):
        # Load MNIST model
        self.model = tf.keras.models.load_model('models/mnist_model.h5')
        (_, _), (self.x_test, _) = tf.keras.datasets.mnist.load_data()
        self.x_test = self.x_test / 255.0
    
    def test_model_inversion(self):
        inverted_images = simulate_model_inversion(self.model, self.x_test, target_class=1, num_samples=10)
        self.assertEqual(inverted_images.shape[0], 10)  # Should reconstruct 10 samples
        self.assertEqual(inverted_images.shape[1:], (28, 28))  # Should match image shape

if __name__ == '__main__':
    unittest.main()
