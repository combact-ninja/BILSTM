# import tensorflow as tf
# from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Concatenate, Lambda
# from tensorflow.keras.initializers import HeNormal
#
#
# class BasicConv(tf.keras.layers.Layer):
#     def __init__(self, out_planes, kernel_size):
#         super(BasicConv, self).__init__()
#         self.conv = Conv2D(
#             out_planes,
#             kernel_size=(kernel_size, kernel_size),
#             strides=(1, 1),
#             padding='same',
#             kernel_initializer=HeNormal(),
#             use_bias=False
#         )
#         self.bn = BatchNormalization(axis=-1, momentum=0.999, epsilon=1e-5)
#
#     def call(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = ReLU()(x)
#         return x
#
#
# class ChannelPool(tf.keras.layers.Layer):
#     def call(self, x):
#         max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
#         avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
#         return Concatenate(axis=-1)([max_pool, avg_pool])
#
#
# class SpatialGate(tf.keras.layers.Layer):
#     def __init__(self):
#         super(SpatialGate, self).__init__()
#         self.compress = ChannelPool()
#         self.spatial = BasicConv(out_planes=1, kernel_size=3)
#
#     def call(self, x):
#         x_compress = self.compress(x)
#         x_out = self.spatial(x_compress)
#         scale = tf.nn.sigmoid(x_out)
#         return x * scale
#
#
# class TripletAttention(tf.keras.layers.Layer):
#     def __init__(self, no_spatial=False):
#         super(TripletAttention, self).__init__()
#         self.ChannelGateH = SpatialGate()
#         self.ChannelGateW = SpatialGate()
#         self.no_spatial = no_spatial
#         if not no_spatial:
#             self.SpatialGate = SpatialGate()
#
#     def call(self, x):
#         # Permute and process with ChannelGateH
#         x_perm1 = tf.transpose(x, perm=[0, 2, 1, 3])  # Swap height and width
#         x_out1 = self.ChannelGateH(x_perm1)
#         x_out1 = tf.transpose(x_out1, perm=[0, 2, 1, 3])  # Revert height and width
#
#         # Permute and process with ChannelGateW
#         x_perm2 = tf.transpose(x, perm=[0, 3, 2, 1])  # Swap height and channels
#         x_out2 = self.ChannelGateW(x_perm2)
#         x_out2 = tf.transpose(x_out2, perm=[0, 3, 2, 1])  # Revert height and channels
#
#         # Combine with SpatialGate if not disabled
#         if not self.no_spatial:
#             x_out = self.SpatialGate(x)
#             x_out = (x_out + x_out1 + x_out2) / 3.0
#         else:
#             x_out = (x_out1 + x_out2) / 2.0
#
#         return x_out



import tensorflow as tf
from keras import Input, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Concatenate
from tensorflow.keras.initializers import HeNormal


class GatedChannelTransform(tf.keras.layers.Layer):
    def __init__(self, channels, epsilon=1e-5, **kwargs):
        super(GatedChannelTransform, self).__init__(**kwargs)
        self.channels = channels
        self.epsilon = epsilon

    def build(self, input_shape):
        # Correct weight shapes: (1, 1, 1, C) for broadcasting along channels
        self.alpha = self.add_weight(
            shape=(1, 1, 1, self.channels), initializer='ones', trainable=True, name='alpha')
        self.gamma = self.add_weight(
            shape=(1, 1, 1, self.channels), initializer='ones', trainable=True, name='gamma')
        self.beta = self.add_weight(
            shape=(1, 1, 1, self.channels), initializer='zeros', trainable=True, name='beta')

    def call(self, x):
        # Reduce over spatial dimensions [1, 2] (height and width)
        embedding = tf.sqrt(tf.reduce_sum(tf.square(x), axis=[1, 2], keepdims=True) + self.epsilon) * self.alpha

        # Normalize across the channel dimension
        norm = self.gamma / tf.sqrt(tf.reduce_mean(tf.square(embedding), axis=-1, keepdims=True) + self.epsilon)

        # Gating mechanism
        gate = 1.0 + tf.tanh(embedding * norm + self.beta)
        return x * gate  # Apply gating


class BasicConv(tf.keras.layers.Layer):
    def __init__(self, out_planes, kernel_size):
        super(BasicConv, self).__init__()
        self.conv = Conv2D(
            out_planes,
            kernel_size=(kernel_size, kernel_size),
            strides=(1, 1),
            padding='same',
            kernel_initializer=HeNormal(),
            use_bias=False
        )
        self.bn = BatchNormalization(axis=-1, momentum=0.999, epsilon=1e-5)

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = ReLU()(x)
        return x


class ChannelPool(tf.keras.layers.Layer):
    def call(self, x):
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        return Concatenate(axis=-1)([max_pool, avg_pool])


class SpatialGate(tf.keras.layers.Layer):
    def __init__(self, channels=None):
        super(SpatialGate, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(out_planes=1, kernel_size=3)
        if channels is not None:
            self.gated_transform = GatedChannelTransform(channels=channels)

    def call(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = tf.nn.sigmoid(x_out)

        # Apply gating if Gated Channel Transform is present
        if hasattr(self, 'gated_transform'):
            x = self.gated_transform(x)
        return x * scale


class TripletAttention(tf.keras.layers.Layer):
    def __init__(self, no_spatial=False, channels=None):
        super(TripletAttention, self).__init__()
        self.ChannelGateH = SpatialGate(channels=channels)
        self.ChannelGateW = SpatialGate(channels=channels)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate(channels=channels)
        if channels is not None:
            self.gated_transform = GatedChannelTransform(channels=channels)

    def call(self, x):
        # Permute and process with ChannelGateH
        x_perm1 = tf.transpose(x, perm=[0, 2, 1, 3])  # Swap height and width
        x_out1 = self.ChannelGateH(x_perm1)
        x_out1 = tf.transpose(x_out1, perm=[0, 2, 1, 3])  # Revert height and width

        # Permute and process with ChannelGateW
        x_perm2 = tf.transpose(x, perm=[0, 3, 2, 1])  # Swap height and channels
        x_out2 = self.ChannelGateW(x_perm2)
        x_out2 = tf.transpose(x_out2, perm=[0, 3, 2, 1])  # Revert height and channels

        # Combine with SpatialGate if not disabled
        if not self.no_spatial:
            x_out = self.SpatialGate(x)
            x_out = (x_out + x_out1 + x_out2) / 3.0
        else:
            x_out = (x_out1 + x_out2) / 2.0

        # Apply Gated Transform to the aggregate output
        if hasattr(self, 'gated_transform'):
            x_out = self.gated_transform(x_out)

        return x_out







input_shape = (64, 64, 3)  # Example input
inputs = Input(shape=input_shape)
triplet_attention = TripletAttention(no_spatial=False, channels=3)  # Set channels to 3 for RGB images
attention = triplet_attention(inputs)
outputs = tf.keras.layers.Conv2D(10, (3, 3), activation='relu')(attention)

model = Model(inputs, outputs)

# Plot the model
tf.keras.utils.plot_model(model, to_file='triplet_attention_model.png', show_shapes=True, expand_nested=True)



# ----------------------------------- model1-----------------------------
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. Simulate Input Data
# Generate synthetic input data and labels (example with random data)
num_samples = 1000
input_shape = (120, 120, 5)  # Input shape (120, 120, 5 channels)
num_classes = 4

X_data = np.random.rand(num_samples, *input_shape).astype(np.float32)  # Random input images
y_data = np.random.randint(0, num_classes, size=(num_samples,))  # Random labels for 4 classes

# Split into train-test data
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# 2. Define DeepCNN Model for Feature Extraction
def build_deepcnn(input_shape):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    feature_output = Dense(128, activation='relu', name="feature_output")(x)  # Feature layer

    output = Dense(num_classes, activation='softmax', name="cnn_output")(feature_output)
    model = Model(inputs, [feature_output, output], name="DeepCNN")
    return model

# Build DeepCNN model
cnn_model = build_deepcnn(input_shape)
cnn_model.summary()

# Compile CNN model (optional - for CNN evaluation itself)
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 3. Train DeepCNN to Get Features
# Train the CNN for a few epochs to extract features
print("Training the DeepCNN model...")
history = cnn_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Extract CNN features for train and test data
print("Extracting features using DeepCNN...")
feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.get_layer("feature_output").output)

X_train_features = feature_extractor.predict(X_train)
X_test_features = feature_extractor.predict(X_test)

# 4. XGBoost Classifier with CNN Features
print("Training XGBoost Classifier on extracted CNN features...")
xgb_clf = XGBClassifier(objective='multi:softmax', num_class=num_classes, eval_metric='mlogloss')
xgb_clf.fit(X_train_features, y_train)

# Predictions using XGBoost
y_pred = xgb_clf.predict(X_test_features)
xgb_accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Classifier Accuracy: {xgb_accuracy * 100:.2f}%")

# 5. Ensemble Predictions (Optional Adaptive Combination)
# Combine CNN and XGBoost predictions adaptively
cnn_preds = cnn_model.predict(X_test)[1]  # CNN softmax predictions
xgb_preds = tf.keras.utils.to_categorical(y_pred, num_classes)  # Convert XGBoost output to one-hot

# Weighted averaging between CNN and XGBoost
alpha = 0.5  # Weight for CNN; 1-alpha for XGBoost
ensemble_preds = alpha * cnn_preds + (1 - alpha) * xgb_preds
ensemble_final = np.argmax(ensemble_preds, axis=1)

# Final ensemble accuracy
ensemble_accuracy = accuracy_score(y_test, ensemble_final)
print(f"Ensemble Model Accuracy: {ensemble_accuracy * 100:.2f}%")

# 6. Visualize Training Loss and Accuracy for CNN
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('CNN Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('CNN Training and Validation Accuracy')
plt.legend()
plt.show()



# -----------------------------------------------------------------------
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

# 1. Generate Synthetic Input Data and Labels
num_samples = 1000
input_shape = (120, 120, 5)
num_classes = 4

X_data = np.random.rand(num_samples, *input_shape).astype(np.float32)
y_data = np.random.randint(0, num_classes, size=(num_samples,))

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# 2. Define DeepCNN for Feature Extraction and Classification
def build_deepcnn(input_shape):
    inputs = Input(shape=input_shape)
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    feature_output = Dense(128, activation='relu', name="feature_output")(x)  # Extracted feature vector
    output = Dense(num_classes, activation='softmax', name="cnn_output")(feature_output)
    
    model = Model(inputs, [feature_output, output], name="DeepCNN")
    return model

cnn_model = build_deepcnn(input_shape)
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(cnn_model.summary())

# 3. Train the CNN to Extract Features
print("Training CNN for feature extraction...")
cnn_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=2)

# Feature extraction
feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.get_layer("feature_output").output)
X_train_features = feature_extractor.predict(X_train)
X_test_features = feature_extractor.predict(X_test)

# 4. Train XGBoost on CNN-Extracted Features
print("Training XGBoost...")
xgb_model = XGBClassifier(objective='multi:softmax', num_class=num_classes, eval_metric='mlogloss')
xgb_model.fit(X_train_features, y_train)
xgb_preds_proba = xgb_model.predict_proba(X_test_features)  # Probabilities

# 5. Predict Probabilities Using CNN
cnn_preds_proba = cnn_model.predict(X_test)[1]

# 6. Adaptive Weight Calculation Based on Log Loss
cnn_logloss = log_loss(y_test, cnn_preds_proba)
xgb_logloss = log_loss(y_test, xgb_preds_proba)

# Invert log loss to get performance-based weights
cnn_weight = 1 / cnn_logloss
xgb_weight = 1 / xgb_logloss
sum_weights = cnn_weight + xgb_weight

cnn_weight /= sum_weights
xgb_weight /= sum_weights

print(f"Adaptive Weights: CNN={cnn_weight:.2f}, XGBoost={xgb_weight:.2f}")

# 7. Adaptive Ensemble Prediction (Weighted Averaging)
ensemble_proba = cnn_weight * cnn_preds_proba + xgb_weight * xgb_preds_proba
ensemble_preds = np.argmax(ensemble_proba, axis=1)

# Final Evaluation
accuracy = accuracy_score(y_test, ensemble_preds)
print(f"Adaptive Ensemble Accuracy: {accuracy * 100:.2f}%")

