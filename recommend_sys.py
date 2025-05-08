import tensorflow as tf
from tensorflow.keras.layers import Layer

class BahdanauAttention(Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, hidden_states, decoder_hidden):
        # hidden_states: (batch_size, seq_len, hidden_size)
        # decoder_hidden: (batch_size, hidden_size)
        decoder_hidden = tf.expand_dims(decoder_hidden, 1)  # (batch_size, 1, hidden_size)
        score = self.V(tf.nn.tanh(self.W1(hidden_states) + self.W2(decoder_hidden)))
        attention_weights = tf.nn.softmax(score, axis=1)  # (batch_size, seq_len, 1)
        context_vector = attention_weights * hidden_states
        context_vector = tf.reduce_sum(context_vector, axis=1)  # (batch_size, hidden_size)
        return context_vector, attention_weights


class SelfAttention(Layer):
    def __init__(self, d_model):
        super().__init__()
        self.Wq = tf.keras.layers.Dense(d_model)
        self.Wk = tf.keras.layers.Dense(d_model)
        self.Wv = tf.keras.layers.Dense(d_model)

    def call(self, x):
        # x: (batch_size, seq_len, features)
        Q = self.Wq(x)  # (batch_size, seq_len, d_model)
        K = self.Wk(x)
        V = self.Wv(x)

        matmul_qk = tf.matmul(Q, K, transpose_b=True)  # (batch_size, seq_len, seq_len)
        dk = tf.cast(tf.shape(K)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (batch_size, seq_len, seq_len)
        output = tf.matmul(attention_weights, V)  # (batch_size, seq_len, d_model)
        return output, attention_weights




# Assume roberta_embeddings is output from pretrained RoBERTa
lstm_out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(roberta_embeddings)

# For Bahdanau Attention
context_vector, attn_weights = BahdanauAttention(64)(lstm_out, lstm_out[:, -1, :])  # decoder_hidden = last hidden

# OR For Self-Attention
# self_attn_out, attn_weights = SelfAttention(128)(lstm_out)
# context_vector = tf.reduce_mean(self_attn_out, axis=1)  # or use global average/max pooling

# Final output
output = tf.keras.layers.Dense(64, activation='relu')(context_vector)
output = tf.keras.layers.Dense(num_classes, activation='softmax')(output)
