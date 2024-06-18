import tensorflow as tf
from tensorflow.keras import layers, models

def build_generator(noise_dim):
    model = models.Sequential()
    
    # Dense layer to scale up the noise vector
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    # Reshape to start the convolutional stack
    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)
    
    # Transpose Convolutional Layer 1
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    # Transpose Convolutional Layer 2
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    # Transpose Convolutional Layer 3
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)
    
    return model

# Set the dimensionality of the noise vector
noise_dim = 100

# Create the generator model
generator = build_generator(noise_dim)

# Display the generator model summary
generator.summary()
