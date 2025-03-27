# src/model_inversion.py

import numpy as np
import tensorflow as tf

def simulate_model_inversion(model, x_train, target_class=1, num_samples=100):
    """
    Simulate model inversion by trying to reconstruct training data for a given target class.
    
    Args:
    - model: The trained model.
    - x_train: The training data (images).
    - target_class: The class for which we are reconstructing data.
    - num_samples: The number of samples to reconstruct.
    
    Returns:
    - inverted_images: The reconstructed images.
    """
    target_indices = np.where(np.argmax(model.predict(x_train), axis=1) == target_class)[0]
    target_images = x_train[target_indices][:num_samples]
    
    inverted_images = []
    for image in target_images:
        image = tf.Variable(image)
        with tf.GradientTape() as tape:
            tape.watch(image)
            prediction = model(tf.expand_dims(image, axis=0))
            loss = tf.keras.losses.sparse_categorical_crossentropy(np.full((1,), target_class), prediction)
        gradient = tape.gradient(loss, image)
        inverted_image = image - 0.1 * gradient.numpy().squeeze()  # Update image using gradient descent
        inverted_images.append(inverted_image)
    
    return np.array(inverted_images)
