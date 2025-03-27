import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def simulate_data_poisoning():
    """
    Simulates data poisoning by introducing incorrect labels into the training set.
    """
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Preprocess data
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    # Inject poisoned data (e.g., flip labels of 100 random samples)
    poisoned_data = np.copy(x_train[:100])
    poisoned_labels = np.copy(y_train[:100])
    
    # Flip labels for 100 random samples to simulate poisoning
    poisoned_labels[:50] = np.random.randint(0, 10, 50)
    
    # Add poisoned data to the training set
    x_train = np.concatenate([x_train, poisoned_data], axis=0)
    y_train = np.concatenate([y_train, poisoned_labels], axis=0)

    # Train a model on this poisoned data
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5)

    # Evaluate the model's performance after training with poisoned data
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Accuracy after training with poisoned data: {accuracy:.2f}")
    
    # Return the trained model
    return model
