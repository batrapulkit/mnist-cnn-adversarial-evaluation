# tests/test_data_poisoning.py

import unittest
import tensorflow as tf
from src.data_poisoning import simulate_data_poisoning

class TestDataPoisoning(unittest.TestCase):

    def setUp(self):
        # Load MNIST data
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()
        self.x_train = self.x_train / 255.0
        self.x_test = self.x_test / 255.0
    
    def test_data_poisoning(self):
        poison_ratio = 0.1
        poisoned_x_train, poisoned_y_train = simulate_data_poisoning(self.x_train, self.y_train, poison_ratio)
        self.assertEqual(len(poisoned_x_train), len(self.x_train) + int(len(self.x_train) * poison_ratio))
        self.assertEqual(len(poisoned_y_train), len(self.y_train) + int(len(self.y_train) * poison_ratio))

if __name__ == '__main__':
    unittest.main()
