import numpy as np
import tensorflow as tf
from keras.layers import Concatenate
from tensorflow.keras.layers import Layer, LSTM, Bidirectional, Dropout, Dense, Input, Flatten, ELU, LeakyReLU
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import keras.backend as K
import plotly.graph_objects as go

class GraphAttentionLayer(Layer):
    def __init__(self, in_features, out_features, n_heads, concat=True, dropout=0.4, leaky_relu_slope=0.2):
        super(GraphAttentionLayer, self).__init__()

        self.n_heads = n_heads
        self.concat = concat
        self.out_features = out_features if concat else out_features // n_heads
        self.dropout = dropout

        self.W = self.add_weight(
            shape=(in_features, self.out_features * self.n_heads),
            initializer='glorot_uniform',
            trainable=True
        )

        self.a = self.add_weight(
            shape=(self.n_heads, 2 * self.out_features, 1),
            initializer='glorot_uniform',
            trainable=True
        )

        self.leaky_relu = LeakyReLU(leaky_relu_slope)
        self.softmax = tf.keras.layers.Softmax(axis=-1)
        self.dropout_layer = Dropout(dropout)

    def call(self, h, training=False):
        batch_size, n_nodes, in_feat = tf.shape(h)[0], tf.shape(h)[1], tf.shape(h)[2]

        # Create adjacency matrix dynamically using cosine similarity
        h_norm = tf.nn.l2_normalize(h, axis=-1)
        adj_mat = tf.matmul(h_norm, h_norm, transpose_b=True)
        adj_mat = tf.where(adj_mat > 0.5, 1.0, 0.0)  # Thresholding for connectivity

        h_transformed = tf.matmul(h, self.W)  # (batch, n_nodes, n_heads * out_features)
        h_transformed = self.dropout_layer(h_transformed, training=training)

        h_transformed = tf.reshape(h_transformed, (batch_size, n_nodes, self.n_heads, self.out_features))
        h_transformed = tf.transpose(h_transformed, perm=[0, 2, 1, 3])  # (batch, n_heads, n_nodes, out_features)

        source_scores = tf.einsum("bhnd,hdf->bhnf", h_transformed, self.a[:, :self.out_features, :])
        target_scores = tf.einsum("bhnd,hdf->bhnf", h_transformed, self.a[:, self.out_features:, :])

        e = source_scores + tf.transpose(target_scores, perm=[0, 1, 3, 2])
        e = self.leaky_relu(e)

        mask = tf.where(tf.expand_dims(adj_mat, 1) > 0, e, tf.fill(tf.shape(e), -1e9))
        attention = self.softmax(mask)
        attention = self.dropout_layer(attention, training=training)

        h_prime = tf.einsum("bhij,bhjd->bhid", attention, h_transformed)

        if self.concat:
            h_prime = tf.reshape(h_prime, (batch_size, n_nodes, self.n_heads * self.out_features))
        else:
            h_prime = tf.reduce_mean(h_prime, axis=1)

        return h_prime


class GAT(Layer):
    def __init__(self, in_features, n_hidden, n_heads, num_classes, concat=True, dropout=0.4, leaky_relu_slope=0.2):
        super(GAT, self).__init__()

        self.gat1 = GraphAttentionLayer(in_features, n_hidden, n_heads, concat, dropout, leaky_relu_slope)
        self.elu = ELU()
        self.gat2 = GraphAttentionLayer(n_hidden * n_heads if concat else n_hidden, num_classes, 1, False, dropout,
                                        leaky_relu_slope)
        self.softmax = tf.keras.layers.Activation('softmax')

    def call(self, input_tensor, training=False):
        x = self.gat1(input_tensor, training=training)
        x = self.elu(x)
        x = self.gat2(x, training=training)
        return self.softmax(x)


def Bi_LSTM(xtr, xtest, ytr, ytest):
    xtr = np.expand_dims(xtr, axis=-1)
    xtest = np.expand_dims(xtest, axis=-1)
    input_layer = Input(shape=(xtr.shape[1], xtr.shape[2]))

    bilstm_layer = Bidirectional(LSTM(8, activation='relu', return_sequences=True))(input_layer)
    gat_layer = GAT(16, 8, 2, 8)(bilstm_layer, training=True)
    gat_out = Flatten()(gat_layer)

    dense_layer = Dense(8, activation='relu')(gat_out)
    out_layer = Dense(2, activation='softmax')(dense_layer)

    model = Model(inputs=input_layer, outputs=out_layer)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(xtr, ytr, epochs=10, batch_size=16)



def DIS_BILSTM_1(xtrain, xtest, ytrain, ytest,):
    xtrain = np.expand_dims(xtrain, axis=-1)
    xtest = np.expand_dims(xtest, axis=-1)
    ytr = tf.keras.utils.to_categorical(ytrain)
    in_layer = Input(shape=(xtrain.shape[1:]))
    # ---------
    bilstm_layer = Bidirectional(LSTM(32, activation='relu', return_sequences=True))(in_layer)
    gat_layer = GAT(bilstm_layer.shape[-1], 8, 2, 8)(bilstm_layer, training=True)
    branch1 = Bidirectional(LSTM(16, activation='relu', return_sequences=False))(gat_layer)

    # ----------
    bilstm_layer = Bidirectional(LSTM(32, activation='relu', return_sequences=True))(in_layer)
    gat_layer = GAT(bilstm_layer.shape[-1], 8, 2, 8)(bilstm_layer, training=True)
    branch2 = Bidirectional(LSTM(16, activation='relu', return_sequences=False))(gat_layer)

    combined = Concatenate()([branch1, branch2])
    Dense_ = Dense(16, activation='relu', )(combined)
    Dense_ = Dense(8, activation='relu', )(Dense_)
    out_layer = Dense(ytr.shape[1], 'softmax')(Dense_)
    model = Model(inputs=in_layer, outputs=out_layer)
    model.compile('adam', 'categorical_crossentropy', 'accuracy')
    model.fit(xtrain, ytr, epochs=2)

    # Digit twin
    X_twin = xtrain + model.predict(xtrain).reshape(-1)
    import plotly.graph_objects as go

# Custom Hybrid Activation
def tanh_activation(x):
    return K.tanh(x)

def leaky_relu(x, alpha=0.01):
    return K.maximum(alpha * x, x)

def Hybrid_activation(x):
    tanh_out = tanh_activation(x)
    leaky_relu_out = leaky_relu(x)
    return (tanh_out + leaky_relu_out) / 2

# def plot_digital_twin(X_twin):
#     fig = go.Figure()
#     for i in range(min(10, len(X_twin))):
#         fig.add_trace(go.Scatter(y=X_twin[i], mode='lines', name=f'Sample {i + 1}'))
#     fig.update_layout(title='Digital Twin Representation', xaxis_title='Time Steps', yaxis_title='Feature Values')
#     fig.show()


def plot_digital_twin(X_twin, save_path="digital_twin_plot.png"):
    fig = go.Figure()
    for i in range(min(10, len(X_twin))):
        fig.add_trace(go.Scatter(y=X_twin[i], mode='lines', name=f'Sample {i + 1}'))

    fig.update_layout(title='Digital Twin Representation', xaxis_title='Time Steps', yaxis_title='Feature Values')

    # Save as image
    fig.write_image(save_path)
    fig.show()



# Define BiLSTM-GAT Model
def DIS_BILSTM(xtrain, xtest, ytrain, ytest):
    xtrain = np.expand_dims(xtrain, axis=-1)
    xtest = np.expand_dims(xtest, axis=-1)
    ytr = tf.keras.utils.to_categorical(ytrain)
    in_layer = Input(shape=(xtrain.shape[1:]))

    bilstm_layer = Bidirectional(LSTM(32, activation=Hybrid_activation, return_sequences=True))(in_layer)
    gat_layer = GAT(bilstm_layer.shape[-1], 8, 2, 8)(bilstm_layer, training=True)
    branch1 = Bidirectional(LSTM(16, activation=Hybrid_activation, return_sequences=False))(gat_layer)

    bilstm_layer2 = Bidirectional(LSTM(32, activation=Hybrid_activation, return_sequences=True))(in_layer)
    gat_layer2 = GAT(bilstm_layer2.shape[-1], 8, 2, 8)(bilstm_layer2, training=True)
    branch2 = Bidirectional(LSTM(16, activation=Hybrid_activation, return_sequences=False))(gat_layer2)

    combined = Concatenate()([branch1, branch2])
    Dense_ = Dense(16, activation=Hybrid_activation)(combined)
    Dense_ = Dense(8, activation=Hybrid_activation)(Dense_)
    out_layer = Dense(ytr.shape[1], activation='softmax')(Dense_)
    model = Model(inputs=in_layer, outputs=out_layer)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(xtrain, ytr, epochs=2)

    # Digital Twin
    X_twin = xtrain + model.predict(xtrain).reshape(-1)
    plot_digital_twin(X_twin)


from keras.utils.generic_utils import get_custom_objects

get_custom_objects().update({"Hybrid_activation": Hybrid_activation})

# Simulated Data
x = np.random.uniform(0.213, 0.94334, (200, 290))
y = np.random.randint(0, 2, (200))

xtr, xtest, ytr, ytest = train_test_split(x, y, train_size=0.8)

# Bi_LSTM(xtr, xtest, ytr, ytest)
DIS_BILSTM(xtr, xtest, ytr, ytest)
print('hi')
