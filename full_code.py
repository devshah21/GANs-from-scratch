import tensorflow as tf


class GAN(tf.keras.Model):
    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        
    def compile(self, g_optimizer, d_optimizer, loss_fn):
        super(GAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, self.generator.input_shape[-1]])
        
        # Train the discriminator
        with tf.GradientTape() as d_tape:
            generated_images = self.generator(noise, training=True)
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            d_loss_real = self.loss_fn(tf.ones_like(real_output), real_output)
            d_loss_fake = self.loss_fn(tf.zeros_like(fake_output), fake_output)
            d_loss = d_loss_real + d_loss_fake

        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        
        # Train the generator
        with tf.GradientTape() as g_tape:
            generated_images = self.generator(noise, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            g_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)

        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))

        return {"d_loss": d_loss, "g_loss": g_loss}

# Set the dimensionality of the noise vector
noise_dim = 100

from discriminator import *
from generator import *


# Build the generator and discriminator
generator = build_generator(noise_dim)
discriminator = build_discriminator()

# Create the GAN model
gan = GAN(generator, discriminator)

# Compile the GAN model
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
gan.compile(generator_optimizer, discriminator_optimizer, loss_fn)

# Training the GAN
import numpy as np

# Load and preprocess the dataset (MNIST in this case)
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = (train_images - 127.5) / 127.5  # Normalize to [-1, 1]
train_images = np.expand_dims(train_images, axis=-1)

batch_size = 256
dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(60000).batch(batch_size)

# Train the GAN
epochs = 50

for epoch in range(epochs):
    for real_images in dataset:
        gan.train_step(real_images)
    
    # Generate and save images after each epoch (optional)
    noise = tf.random.normal([16, noise_dim])
    generated_images = generator(noise, training=False)
    # Save or display the generated images
