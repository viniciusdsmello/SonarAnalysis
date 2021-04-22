# -*- coding: utf-8 -*-
"""
**Author:** Vinícius Mello<br>
**Date created:** 2021/04/14<br>
**Last modified:** -- <br>
**Description:** Implementation of Variational Autoencoders on Tensorflow.
"""

import os
import logging
import numpy as np
import tensorflow as tf
from typing import List, Union

tfk = tf.keras
tfkl = tfk.layers

"""# Building the model """


def kl_div(z_mean, z_log_var, reduce_on_batch=True):
    kl_divergence = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    kl_divergence = tf.reduce_sum(kl_divergence, axis=1)
    
    if reduce_on_batch:
        kl_divergence = tf.reduce_mean(kl_divergence)
    
    return kl_divergence


class Sampling(tf.keras.layers.Layer):
    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, inputs):
        """
        Reparameterization trick instead of sampling from Q(z|X), sample eps = N(0,I)

        z = z_mean + sqrt(var)*eps

        Args:
            inputs (Tuple): Tuple with mean and variance of a Gaussian Distribution

        Returns:
            np.ndarray: Returns a sample from the parametrized Gaussian Distribution
        """
        mean, log_var = inputs

        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]

        epsilon = tfk.backend.random_normal(shape=(batch, dim))
        return mean + tf.exp(0.5 * log_var) * epsilon


"""## Creating an Encoder

"""


class Encoder(tf.keras.Model):
    def __init__(self, intermediate_dim: Union[int, List[int]], latent_dim: int, name: str = 'encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        # build encoder model

        if isinstance(intermediate_dim, int):
            self.intermediate_dim = list(intermediate_dim)
        else:
            self.intermediate_dim = intermediate_dim

        self.encoder_intermediate = tfk.Sequential([
            tfkl.Dense(
                dim,
                activation=tf.nn.relu,
                name=f'encoder_intermediate_{i}'
            ) for i, dim in enumerate(self.intermediate_dim)
        ])

        self.z_mean = tfkl.Dense(latent_dim,
                                 name='z_mean')
        self.z_log_var = tfkl.Dense(latent_dim,
                                    name='z_log_var')

        self.sampling = Sampling(name='encoder_sampling')

    def sample(self, inputs):
        """Sample Q(z|x) distribution

        Args:
            inputs (Tuple): Tuple with mean and variance of a Gaussian Distribution

        Returns:
            np.ndarray: Returns a sample vector of the Latent Random Variable
        """
        z_sample = self.sampling(inputs)
        return z_sample

    def call(self, inputs, training=False):
        """
        input -> y -> z_mean
        input -> y -> z_log_var
        (z_mean, z_log_var) -> z    q(z|X)

        Args:
            inputs (np.ndarray): 

        Returns:
            np.ndarray: Returns a sample vector of the Latent Random Variable
        """
        y = self.encoder_intermediate(inputs, training=training)

        z_mean = self.z_mean(y, training=training)
        z_log_var = self.z_log_var(y, training=training)
        z = self.sample((z_mean, z_log_var))

        return z_mean, z_log_var, z


"""## Creating a Decoder"""


class Decoder(tf.keras.Model):
    def __init__(self, intermediate_dim, original_dim, name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)

        if isinstance(intermediate_dim, int):
            self.intermediate_dim = list(intermediate_dim)[::-1]
        else:
            self.intermediate_dim = intermediate_dim

        # build decoder model
        self.decoder_intermediate = tfk.Sequential([
            tfkl.Dense(
                dim,
                activation=tf.nn.relu,
                name=f'decoder_intermediate_{i}'
            ) for i, dim in enumerate(self.intermediate_dim)
        ])

        self.output_mean = tfkl.Dense(original_dim,
                                      activation='sigmoid',
                                      name='output_mean')
        self.output_log_var = tfkl.Dense(original_dim,
                                         activation='softplus',
                                         name='output_log_var')

        self.sampling = Sampling(name='decoder_sampling')

    def sample(self, inputs):
        """Sample P(x|z) distribution

        Args:
            inputs (Tuple): Tuple with mean and variance of a Gaussian Distribution

        Returns:
            np.ndarray: Returns a sample vector of the Output Random Variable
        """
        z_sample = self.sampling(inputs)
        return z_sample

    def call(self, latent_inputs, training=False):
        # latent_inputs -> y -> (output_mean, output_log_var)
        # (output_mean, output_log_var) -> x  ~  p(x|z)
        y = self.decoder_intermediate(latent_inputs, training=training)

        output_mean = self.output_mean(y, training=training)
        output_log_var = self.output_log_var(y, training=training)

        output = self.sample((output_mean, output_log_var))
        return output_mean, output_log_var, output


"""## Creating a VAE Model"""


def negative_expected_log_likelihood(x, mu, log_var, sum_on_batch=True):
    """
    Args:
      x:
      mu: Mean of the Gaussian Distribution
      log_var: Variance Log of the Gaussian Distribution
    Returns:
      Returns the negative expected log likelihood
    """
    x_power = -0.5 * tf.square(x - mu) / tf.exp(2 * log_var)
    log_likelihood = -0.5 * (log_var + np.log(2 * np.pi)) + x_power

    if sum_on_batch:
        log_likelihood = tfk.backend.sum(tfk.backend.batch_flatten(log_likelihood), axis=-1)
    else:
        log_likelihood = tfk.backend.sum(log_likelihood, axis=-1)

    return -log_likelihood


class VariationalAutoencoder(tf.keras.Model):
    def __init__(self,
                 encoder: Union[tf.keras.Model, tf.keras.Sequential],
                 decoder: Union[tf.keras.Model, tf.keras.Sequential],
                 latent_dim: int,
                 name='variational_autoencoder',
                 **kwargs):
        super(VariationalAutoencoder, self).__init__(name=name, **kwargs)

        self.encoder = encoder
        self.decoder = decoder

        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")

    @property
    def metrics(self):
        return [
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.total_loss_tracker
        ]

    def call(self, inputs):
        x = inputs
        z_mean, z_log_var, z = self.encoder(x)
        output_mean, output_log_var, output = self.decoder(z)
        return output_mean, output_log_var

    def test_step(self, data):
        # Unpack the data
        x, _ = data

        z_mean, z_log_var, z = self.encoder(x, training=False)
        output_mean, output_log_var, output = self.decoder(z, training=False)
        reconstruction = output

        reconstruction_loss = negative_expected_log_likelihood(data,
                                                               output_mean,
                                                               output_log_var)

        # Compute the loss value
        # (the loss function is configured in `compile()`)
        # self.compiled_loss(x, reconstruction, regularization_losses=self.losses)

        kl_loss = kl_div(z_mean, z_log_var)

        total_loss = reconstruction_loss + kl_loss

        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.total_loss_tracker.update_state(total_loss)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        x, _ = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x, training=True)
            output_mean, output_log_var, output = self.decoder(z, training=True)
            reconstruction = output

            reconstruction_loss = negative_expected_log_likelihood(data,
                                                                   output_mean,
                                                                   output_log_var)

            # Compute the loss value
            # (the loss function is configured in `compile()`)
            # self.compiled_loss(x, reconstruction, regularization_losses=self.losses)

            # Calculate KL Divergence for all distributionslog_likelihoowd
            kl_loss = kl_div(z_mean, z_log_var)

            total_loss = reconstruction_loss + kl_loss

        # Compute gradients
        grads = tape.gradient(total_loss, self.trainable_weights)

        # Update weights
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.total_loss_tracker.update_state(total_loss)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
