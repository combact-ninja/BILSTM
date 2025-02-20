import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Bidirectional, LSTM, Dense, Lambda
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras import layers


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        self.conv = tf.keras.Sequential([
            layers.Conv1D(ch_out, kernel_size=3, strides=1, padding='same', use_bias=True),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv1D(ch_out, kernel_size=3, strides=1, padding='same', use_bias=True),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

    def call(self, x):
        return self.conv(x)


class SqueezeAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, ch_in, ch_out):
        super(SqueezeAttentionBlock, self).__init__()
        self.avg_pool = layers.AveragePooling1D(pool_size=2)
        self.conv = ConvBlock(ch_in, ch_out)
        self.conv_atten = ConvBlock(ch_in, ch_out)
        self.upsample = layers.UpSampling1D(size=2)

    def call_out(self, x):
        x_res = self.conv(x)
        y = self.avg_pool(x)
        y = self.conv_atten(y)
        y = self.upsample(y)
        return (y * x_res) + y


class SupervisedContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def call(self, features, labels):
        # Normalize features
        features = tf.math.l2_normalize(features, axis=0)

        # Compute similarity matrix
        logits = tf.matmul(features, features, transpose_b=True) / self.temperature

        # Create mask for positive pairs
        labels = tf.reshape(labels, [-1, 1])
        mask = tf.equal(labels, tf.transpose(labels))  # [batch, batch]

        # Convert boolean mask to float
        mask = tf.cast(mask, dtype=tf.float32)
        logits_mask = tf.linalg.set_diag(tf.ones_like(mask), tf.zeros(mask.shape[0]))

        # Apply mask to remove self-contrast
        mask *= logits_mask

        # Compute contrastive loss
        exp_logits = tf.exp(logits) * logits_mask
        log_prob = logits - tf.math.log(tf.reduce_sum(exp_logits, axis=1, keepdims=True) + 1e-8)
        mean_log_prob_pos = tf.reduce_sum(mask * log_prob, axis=1) / (tf.reduce_sum(mask, axis=1) + 1e-8)

        # Final loss
        loss = -tf.reduce_mean(mean_log_prob_pos)
        return loss



class ContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def call(self, y_true, y_pred):
        """
        Args:
            y_true: Binary labels (0 for similar pairs, 1 for dissimilar pairs).
            y_pred: Euclidean distance between feature representations.
        Returns:
            Contrastive loss value.
        """
        # Ensure y_true is a float tensor
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        # Compute squared Euclidean distance
        squared_distance = K.square(y_pred)

        # Compute contrastive loss
        loss = K.mean(
            (1 - y_true) * squared_distance +
            y_true * K.square(K.maximum(self.margin - y_pred, 0))
        )
        return loss

# Define Losses
contrastive_loss = ContrastiveLoss()
classification_loss = tf.keras.losses.CategoricalCrossentropy()

def total_loss(y_true, y_pred):
    loss1 = contrastive_loss(y_pred, tf.argmax(y_true, axis=1))  # Contrastive Loss
    loss2 = classification_loss(y_true, y_pred)  # Classification Loss
    return loss1 + loss2  # Combine both losses

def CNN_BiLSTM(X, Y, num_classes):
    X = X.reshape(X.shape[0], X.shape[1], 1)
    input_layer = Input(shape=(X.shape[1], 1))

    # CNN Layer
    cnn = Conv1D(filters=64, kernel_size=3, activation="relu")(input_layer)

    # BiLSTM Layer
    bilstm = Bidirectional(LSTM(64, return_sequences=False))(cnn)

    # Supervised Contrastive Learning Projection
    projection_head = Dense(128, activation="relu")(bilstm)
    normalized_projection = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(projection_head)

    # Classification Head
    output_layer = Dense(num_classes, activation="softmax", name="classification")(normalized_projection)

    # Create Model
    model = Model(inputs=input_layer, outputs=output_layer)



    # Compile Model
    model.compile(
        optimizer="adam",
        loss=total_loss,
        metrics=["accuracy"]
    )

    model.fit(X, Y, epochs=10, batch_size=32)
    return model

# Data Configurations
num_samples = 1000
num_classes = 10
max_len = 20  # Sequence length
max_vocab = 5000  # Simulated vocabulary size

# Generate Random Input Data (Simulating Tokenized Text)
X = np.random.randint(1, max_vocab, size=(num_samples, max_len))  # Random integers as tokenized input

# Generate Random Labels (10 classes, one-hot encoded)
y = np.random.randint(0, num_classes, size=(num_samples,))
y_one_hot = to_categorical(y, num_classes)
CNN_BiLSTM(X, y_one_hot, num_classes)
