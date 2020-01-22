
def vae_loss(y_true, y_pred):
    reconstruction_loss = mse(y_true, y_pred)
    reconstruction_loss *= original_dim
    z_mean = vae.get_layer('encoder').get_layer('z_mean').output
    z_log_var = vae.get_layer('encoder').get_layer('z_log_var').output
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    return K.mean(reconstruction_loss + kl_loss)
