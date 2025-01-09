import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('ML-EdgeIIoT-dataset.csv', low_memory=False)
data = data.select_dtypes(include=[np.number])
data = data.fillna(0)

X = data.drop(['Attack_label'], axis=1, errors='ignore')
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
input_dim = X_scaled.shape[1]
print(f"Number of features: {input_dim}")

generator = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(input_dim,)),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(256),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(input_dim, activation='sigmoid')
])
discriminator = tf.keras.Sequential([
    tf.keras.layers.Dense(256, input_shape=(input_dim,)),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(128),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

@tf.function
def train_step(real_samples, batch_size):
    noise = tf.random.normal([batch_size, input_dim])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_samples = generator(noise, training=True)
        
        real_output = discriminator(real_samples, training=True)
        fake_output = discriminator(generated_samples, training=True)
        
        gen_loss = tf.reduce_mean(tf.math.log(1e-10 + 1 - fake_output))
        disc_loss = -tf.reduce_mean(tf.math.log(1e-10 + real_output) + tf.math.log(1e-10 + 1 - fake_output))
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss

epochs = 800
batch_size = 64
g_losses = []
d_losses = []

for epoch in range(epochs):
    idx = np.random.randint(0, X_scaled.shape[0], batch_size)
    real_batch = X_scaled[idx]
    real_batch = tf.convert_to_tensor(real_batch, dtype=tf.float32)
    
    g_loss, d_loss = train_step(real_batch, batch_size)

    g_losses.append(float(g_loss))
    d_losses.append(float(d_loss))
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Generator loss: {float(g_loss):.4f}, Discriminator loss: {float(d_loss):.4f}')

plt.figure(figsize=(10, 5))
plt.plot(g_losses, label='Generator Loss')
plt.plot(d_losses, label='Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Progress')
plt.show()

noise = tf.random.normal([100, input_dim])
synthetic_samples = generator(noise).numpy()