# %% 
import tensorflow as tf
from tensorflow.keras import layers

# Sample data dimensions (1000 samples, 100 timesteps, 3 features)
n_samples = 1000
n_timesteps = 100
n_features = 3

# Generate synthetic data
X = tf.random.normal((n_samples, n_timesteps, n_features))
y = tf.random.uniform((n_samples,), maxval=2, dtype=tf.int32)

# Model architecture
model = tf.keras.Sequential([
    layers.Input(shape=(n_timesteps, n_features)),
    
    # Conv1D with 64 filters and kernel_size=3
    layers.Conv1D(64, 3, activation='relu', padding='same'),
    layers.MaxPooling1D(2),
    
    # Second Conv1D block
    layers.Conv1D(128, 3, activation='relu', padding='same'),
    layers.MaxPooling1D(2),
    
    # Classification head
    layers.GlobalAveragePooling1D(),
    layers.Dense(50, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Use softmax for multi-class
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# %%
