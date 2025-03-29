import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import LeakyReLU, Softmax
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Layer, LeakyReLU, Softmax


class GATLayer(Layer):
    def __init__(self, F_1, K, kernel_regularizer, **kwargs):
        super(GATLayer, self).__init__(**kwargs)
        self.F_1 = F_1
        self.K = K
        self.kernel_regularizer = kernel_regularizer

    def call_(self, inputs):
        X, A = inputs  # X: (batch_size, N, F), A: (batch_size, N, N)

        F = X.shape[-1]  # Number of input features
        N = X.shape[1]  # Number of nodes

        # Weight matrices
        W = self.add_weight(name='W',
                            shape=(F, self.F_1),
                            initializer='glorot_uniform',
                            regularizer=self.kernel_regularizer,
                            trainable=True)

        a = self.add_weight(name='a',
                            shape=(2, self.F_1, 1),
                            initializer='glorot_uniform',
                            regularizer=self.kernel_regularizer,
                            trainable=True)

        # Linear transformation
        x_features = tf.matmul(X, W)  # Shape: (batch_size, N, F_1)

        # Compute attention scores
        att_self = tf.matmul(x_features, a[0])  # Shape: (batch_size, N, 1)
        att_neighbours = tf.matmul(x_features, a[1])  # Shape: (batch_size, N, 1)

        # Fix transpose operation
        att_neighbours = tf.transpose(att_neighbours, perm=[0, 2, 1, 3])  # Shape: (batch_size, 1, N)

        att = att_self + att_neighbours  # Now shape is (batch_size, N, N, 1)
        att = LeakyReLU(alpha=0.2)(att)

        # Fix shape mismatch of mask
        out = np.mean(A, axis=1)  # Shape: (1,)
        adj_matrix = tf.convert_to_tensor(out, dtype=tf.float32)
        adj_matrix = tf.expand_dims(adj_matrix, axis=0)
        adj_matrix = tf.tile(adj_matrix, [tf.shape(X)[0], 1])  # (None, 5625, 5625)
        mask = -1e9 * (1.0 - adj_matrix)  # Shape: (batch_size, N, N)
        mask = tf.expand_dims(mask, axis=-1)  # Shape: (batch_size, N, N, 1)
        reshaped_tensor = tf.reshape(mask, (-1, 75, 75, 1))
        # mask_out = tf.reduce_mean(mask, axis=[0, 1])  # Shape: (1,)
        # out = tf.reshape(mask_out, (5625, 1))

        att_masked = att + reshaped_tensor  # Now both tensors have shape (batch_size, N, N, 1)

        # Normalize attention scores
        dense = Softmax(axis=-2)(att_masked)  # Shape: (batch_size, N, N, 1)

        # Compute output (Fix: Ensure correct shape)
        dense = tf.matmul(tf.squeeze(dense, axis=-1), x_features)  # Shape: (batch_size, N, F_1)
        dense = tf.concat([dense, x_features], axis=-1)
        return dense



# Define Multi-Scale Graph Attention (MSGA)
def MSGA(input_tensor): # None, 75,75, 32
    # Apply GAT at different scales and concatenate the results
    scale1 = Conv2D(16, (3, 3), padding='same', activation='relu')(input_tensor)  # scale one is None, 75,75,16
    scale2 = Conv2D(32, (3, 3), padding='same', activation='relu')(input_tensor)
    scale3 = Conv2D(64, (3, 3), padding='same', activation='relu')(input_tensor)

    # Create adjacency matrices for each scale (4-connectivity adjacency)
    adj_matrix1 = create_adjacency_matrix(scale1.shape[1], scale1.shape[2]) # array 5624, 5624 can I  cretae None, 5624, 5624 that means for each row
    adj_matrix2 = create_adjacency_matrix(scale2.shape[1], scale2.shape[2])
    adj_matrix3 = create_adjacency_matrix(scale3.shape[1], scale3.shape[2])


    # Apply GAT to each scale
    scale1_attention = GATLayer(F_1=8, K=8, kernel_regularizer=l2(5e-4)).call_([scale1, adj_matrix1])
    scale2_attention = GATLayer(F_1=8, K=8, kernel_regularizer=l2(5e-4)).call_([scale2, adj_matrix2])
    scale3_attention = GATLayer(F_1=8, K=8, kernel_regularizer=l2(5e-4)).call_([scale3, adj_matrix3])

    # Concatenate all attention outputs
    concatenated_attention = Concatenate()([scale1_attention, scale2_attention, scale3_attention])

    return concatenated_attention


# Function to create adjacency matrix for graph (4-connectivity)
def create_adjacency_matrix(rows, cols):
    num_nodes = rows * cols
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(rows):
        for j in range(cols):
            node_id = i * cols + j
            if i > 0:  # Up
                adj_matrix[node_id, (i - 1) * cols + j] = 1
            if i < rows - 1:  # Down
                adj_matrix[node_id, (i + 1) * cols + j] = 1
            if j > 0:  # Left
                adj_matrix[node_id, i * cols + (j - 1)] = 1
            if j < cols - 1:  # Right
                adj_matrix[node_id, i * cols + (j + 1)] = 1
    return adj_matrix


# Model Build Function with Graph Attention
def Model_build(xtrain, xtest, ytrain, ytest):
    input_layer = Input(shape=xtrain.shape[1:])

    # Convolutional layers with pooling
    cnn_layer = Conv2D(16, (3, 3), padding='same', activation='relu')(input_layer)
    pool_layer = MaxPooling2D(pool_size=(2, 2))(cnn_layer)
    cnn_layer_ = Conv2D(32, (3, 3), padding='same', activation='relu')(pool_layer)
    pool_layer_ = MaxPooling2D(pool_size=(2, 2))(cnn_layer_)

    # Apply Multi-Scale Graph Attention
    msca_output = MSGA(pool_layer_)

    # Upsampling and additional layers for refinement
    upsampled = UpSampling2D(size=(2, 2))(msca_output)
    upsampled_2 = UpSampling2D(size=(2, 2))(upsampled)

    # Final convolution to output density map
    density_map = Conv2D(1, (1, 1), activation='linear')(upsampled_2)

    # Create the model
    model = Model(inputs=input_layer, outputs=density_map)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    return model


# Preprocessing data
# x = np.load('NPY Files/im_arr.npy')[:10]
x = np.random.uniform(0.3, 53.42, (10, 500, 500, 3))
x = x.astype('float32') / 255.0  # Normalize the data
# y = np.load('NPY Files/Lab_arr.npy')[:10]
y = np.random.randint(150, 1900, (10))

# Resize images to a consistent size
X = [cv2.resize(im, (300, 300)) for im in x]

# Train-test split
xtrain, xtest, ytrain, ytest = train_test_split(np.array(X), y, train_size=0.8)
# Build and train the model
model = Model_build(xtrain, xtest, ytrain, ytest)
# Print model summary
model.summary()
# Training the model
model.fit(xtrain, ytrain, validation_data=(xtest, ytest), batch_size=1, epochs=2)
model.predict(xtest)
