# src/utils.py

import numpy as np
import tensorflow as tf

def generate_adversarial_example(model, image, label, epsilon=0.1):
    """
    Generates an adversarial example using the Fast Gradient Sign Method (FGSM).
    
    Args:
    - model: The trained model.
    - image: The input image to perturb.
    - label: The true label for the image.
    - epsilon: The magnitude of perturbation.
    
    Returns:
    - adversarial_image: The adversarially perturbed image.
    - original_pred: The original prediction.
    - adversarial_pred: The prediction on the adversarial image.
    """
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    label = tf.convert_to_tensor(label, dtype=tf.int64)
    
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    label = tf.expand_dims(label, axis=0)  # Add batch dimension

    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = tf.keras.losses.sparse_categorical_crossentropy(label, prediction)
    
    gradient = tape.gradient(loss, image)
    perturbation = epsilon * tf.sign(gradient)
    adversarial_image = image + perturbation
    adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)  # Ensure valid pixel values
    
    original_pred = tf.argmax(model(image), axis=1)
    adversarial_pred = tf.argmax(model(adversarial_image), axis=1)

    return adversarial_image.numpy().squeeze(), original_pred.numpy(), adversarial_pred.numpy()
