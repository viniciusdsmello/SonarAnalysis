import tensorflow as tf
import numpy as np


class VariationalAutoencoder(tf.keras.Model):
    def __init__(self, original_dim, intermediate_dim, latent_dim, name='variational_autoencoder', **kwargs):
        super(VariationalAutoencoder, self).__init__(name=name, **kwargs)

        self._original_dim = original_dim
        self._intermediate_dim = intermediate_dim
        self._latent_dim = latent_dim

        self.encoder = Encoder(self._intermediate_dim, self._latent_dim)
        self.decoder = Decoder(self._intermediate_dim, self._original_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = -0.5 * tf.reduce_sum(kl_loss)
        self.add_loss(kl_loss)
        return reconstructed


class Sampling(tf.keras.layers.Layer):
    # reparameterization trick
    # instead of sampling from Q(z|X), sample eps = N(0,I)
    # z = z_mean + sqrt(var)*eps
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(tf.keras.layers.Layer):
    def __init__(self, intermediate_dim, latent_dim, name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        # build encoder model
        self.x = tf.keras.layers.Dense(intermediate_dim, activation=tf.nn.relu, name='encoder_intermediate')
        self.z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')
        self.z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')
        self.sampling = Sampling()

    def call(self, inputs):
        y = self.x(inputs)
        z_mean = self.z_mean(y)
        z_log_var = self.z_log_var(y)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(tf.keras.layers.Layer):
    def __init__(self, intermediate_dim, original_dim, name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        # build decoder model
        self.x = tf.keras.layers.Dense(intermediate_dim, activation=tf.nn.relu, name='decoder_intermediate')
        self.outputs = tf.keras.layers.Dense(original_dim, activation=tf.nn.sigmoid, name='decoder_output')

    def call(self, latent_inputs):
        y = self.x(latent_inputs)
        return self.outputs(y)


def main(**kwargs):
    # MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    image_size = x_train.shape[1]
    original_dim = image_size * image_size
    x_train = np.reshape(x_train, [-1, original_dim])
    x_test = np.reshape(x_test, [-1, original_dim])
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # network parameters
    input_shape = (original_dim,)
    intermediate_dim = 512
    batch_size = 128
    latent_dim = 2
    epochs = 50

    model = VariationalAutoencoder(original_dim, intermediate_dim, latent_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer, loss=tf.keras.losses.mse)
    model.fit(x_train, x_train, epochs=3, batch_size=64)


if __name__ == 'main':
    main()
