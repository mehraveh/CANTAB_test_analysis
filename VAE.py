import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, backend as K

df = pd.read_csv('cantab_res_total.csv')
df = df[df.columns[0:18]]
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_vae(original_dim, intermediate_dim=8, latent_dim=2):
    # Encoder model
    original_inputs = tf.keras.Input(shape=(original_dim,), name='encoder_input')
    x = layers.Dense(intermediate_dim, activation='relu')(original_inputs)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = tf.keras.Model(inputs=original_inputs, outputs=[z_mean, z_log_var, z], name='encoder')
    
    # Decoder model
    latent_inputs = tf.keras.Input(shape=(latent_dim,), name='z_sampling')
    x = layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = layers.Dense(original_dim, activation='sigmoid')(x)
    decoder = tf.keras.Model(inputs=latent_inputs, outputs=outputs, name='decoder')
    
    # VAE model
    outputs = decoder(encoder(original_inputs)[2])
    vae = tf.keras.Model(inputs=original_inputs, outputs=outputs, name='vae')
    
    # Loss function
    reconstruction_loss = tf.keras.losses.binary_crossentropy(original_inputs, outputs)
    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl_loss = tf.reduce_mean(kl_loss)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    
    return encoder, decoder, vae

# Example usage:
original_dim = 2  # Number of features in your tabular dataset
vae_encoder, vae_decoder, vae = build_vae(original_dim=original_dim)

# Compile and train
vae.compile(optimizer='Adagrad')
X = df[['sum_correct_sst', 'avg_rt_ast']]
y = df['coin_ratio']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vae.fit(X_train, X_train, epochs=400, batch_size=32)

# To generate new data:
# Assume you have a latent space representation, you can use the decoder
latent_rep = np.array([[0.5, 0.1]])  # Example latent space representation
generated_data = vae_decoder.predict(latent_rep)
print(generated_data)