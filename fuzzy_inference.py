
import tensorflow as tf
import numpy as np
import random
from itertools import product
from tqdm import tqdm
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, MultiHeadAttention, GlobalAveragePooling1D, \
    Dense, Dropout, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

# ----------------------
# Hyperparameters
# ----------------------
n_neurons = 5  # Reduced number of fuzzy rules (neurons)
n_feature = 45  # Reduced number of features
batch_size = 32  # Training batch size
n_femap = 1  # Reduced number of feature maps in fuzzy inference
mu = 0.5  # Fuzzy membership mean
sigma = 0.1  # Fuzzy membership standard deviation

# ----------------------
# Generate fuzzy rules
# ----------------------
out_fRules = np.random.choice([-1.0, 0.0, 1.0], size=(n_neurons, n_feature))

# Convert to tensor
fRules_sigma = K.variable(out_fRules.astype(np.float32))

# Convert to tensor
# fRules_sigma = K.variable(np.array(out_fRules, dtype=np.float32))

# ----------------------
# Define Fuzzy Inference Block
# ----------------------
# ----------------------
# Define Fuzzy Inference Block
# ----------------------
# ----------------------
# Define Fuzzy Inference Block
# ----------------------
class FuzzyInferenceBlock(tf.keras.layers.Layer):
    def __init__(self, output_dim, i_fmap, mu, sigma):  # Remove **kwargs
        super(FuzzyInferenceBlock, self).__init__()
        self.output_dim = output_dim
        self.i_fmap = i_fmap  # Store i_fmap
        self.mu = tf.Variable(tf.random.uniform((output_dim, n_feature), 0, 1) * mu, trainable=True)
        self.sigma = tf.Variable(tf.random.uniform((output_dim, n_feature), 0, 1) * sigma, trainable=True)

    def call_(self, inputs):
        # Ensure inputs match the expected feature size
        inputs = tf.keras.layers.Dense(n_feature, activation="relu")(inputs)  # Adjust feature size

        # Expand dimensions for Gaussian calculation
        aligned_x = tf.expand_dims(inputs, axis=1)  # Shape: (batch_size, 1, n_feature)
        aligned_c = tf.expand_dims(self.mu, axis=0)  # Shape: (1, output_dim, n_feature)
        aligned_s = tf.expand_dims(self.sigma, axis=0)  # Shape: (1, output_dim, n_feature)

        # Compute the Gaussian membership function
        phi = tf.exp(-tf.reduce_sum(tf.square(aligned_x - aligned_c) / (2 * tf.square(aligned_s)), axis=-1))
        return phi


# ----------------------
# Define the Main Model
# ----------------------
def build_model(x_train, y_train):
    input_shape = (45, 1)  # Reduced time series length

    # Input layer
    inputs = Input(shape=x_train.shape[1:])

    # 1D CNN feature extraction
    x = Conv1D(filters=32, kernel_size=3, padding="same", activation="relu")(inputs)
    x = BatchNormalization()(x)

    x = Conv1D(filters=64, kernel_size=3, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)

    # Multi-Head Attention (MHA)
    mha = MultiHeadAttention(num_heads=2, key_dim=32)(x, x)  # Reduced number of heads
    x = tf.keras.layers.Add()([x, mha])  # Residual connection
    x = tf.keras.layers.LayerNormalization()(x)

    # Global Pooling
    x = GlobalAveragePooling1D()(x)

    # Fully connected layers
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(16, activation="relu")(x)
    x = Dropout(0.3)(x)

    # Fuzzy Inference Layer
    fuzzy_outputs = []
    for i in tqdm(range(n_femap)):
        f_block = FuzzyInferenceBlock(output_dim=n_neurons, i_fmap=i, mu=mu, sigma=sigma).call_(x)
        fuzzy_outputs.append(f_block)

    merged_fuzzy = concatenate(fuzzy_outputs, axis=1)

    # Output layer (binary classification)
    outputs = Dense(y_train.shape[1], activation="softmax")(merged_fuzzy)  # Use 'softmax' for multi-class

    # Model creation
    model = Model(inputs, outputs)

    # Compile model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Summary
    model.summary()
    model.fit(x_train, y_train, batch_size=batch_size, epochs=2)  # Adjust epochs for testing

    return model

# ----------------------
# Create Random Data and Train Model
# ----------------------
x_train = np.random.uniform(0.99, 1.363, (1000, 45, 1))  # Reduced size of input
y_train = np.random.randint(0, 7, (1000))  # Example: 7 categories
y_train = tf.keras.utils.to_categorical(y_train)

# Build and train model
model = build_model(x_train, y_train)
