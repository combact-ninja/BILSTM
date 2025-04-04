# import cv2
# import numpy as np
# from skimage import measure, morphology, filters
# import matplotlib.pyplot as plt
# import pydicom
# from glob import glob
#
# def preprocess_image(img):
#     # Normalize image to the range 0 to 1
#     img = img.astype(np.float32) / np.max(img)
#     img = cv2.resize(img, (256, 256))
#     # Apply Gaussian blur to reduce noise and enhance lobes
#     img = cv2.GaussianBlur(img, (5, 5), 0)
#     return img
#
# def segment_lung_region(img):
#     # Adaptive thresholding to handle varying intensity in CT images
#     img_blurred = cv2.GaussianBlur(img, (5, 5), 0)
#     thresh_val = filters.threshold_otsu(img_blurred)
#     binary_img = img_blurred > thresh_val
#
#     # Morphological operations to refine lung boundaries
#     # binary_img = morphology.remove_small_objects(binary_img, min_size=200)
#     # binary_img = morphology.remove_small_holes(binary_img, area_threshold=400)
#
#     # Erode and dilate to refine segmentation (adjust kernel size)
#     kernel = np.ones((3, 3), np.uint8)
#     binary_img = cv2.erode(binary_img.astype(np.uint8), kernel, iterations=1)
#     binary_img = cv2.dilate(binary_img, kernel, iterations=2)
#
#     # Label connected regions to focus on larger lobes
#     labels = measure.label(binary_img)
#     areas = [r.area for r in measure.regionprops(labels)]
#     areas.sort()
#
#     if len(areas) > 2:
#         min_area_threshold = areas[-3]
#         for region in measure.regionprops(labels):
#             if region.area < min_area_threshold:
#                 for coordinates in region.coords:
#                     labels[coordinates[0], coordinates[1]] = 0
#
#     binary_img = labels > 0
#     return binary_img
#
# def center_focus_mask(segmented_lungs, img_shape):
#     # Create a circular mask focusing on the center of the image
#     center_x, center_y = img_shape[1] // 2, img_shape[0] // 2
#     radius = min(img_shape) // 3
#
#     Y, X = np.ogrid[:img_shape[0], :img_shape[1]]
#     dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
#     mask = dist_from_center <= radius
#
#     return segmented_lungs * mask
#
# def visualize_results(preprocessed_img, segmented_lungs, focused_lungs):
#     fig, ax = plt.subplots(1, 3, figsize=(18, 6))
#
#     ax[0].imshow(preprocessed_img, cmap='gray')
#     ax[0].set_title("Preprocessed Image")
#
#     ax[1].imshow(segmented_lungs, cmap='gray')
#     ax[1].set_title("Segmented Lungs")
#
#     ax[2].imshow(focused_lungs, cmap='gray')
#     ax[2].set_title("Central Lobe Segmentation")
#
#     # plt.show()
#     plt.pause(2)
#     plt.close()
#
# # Example usage with DICOM images
# path = 'Lung_Dx-A0001/04-04-2007-NA-Chest-07990/2.000000-5mm-40805'
# all_imgs = glob(path + "/*.dcm")[20:31]
#
# for file in all_imgs:
#     img_arr = pydicom.dcmread(file)
#     img_arr_ = img_arr.pixel_array
#
#     preprocessed_img = preprocess_image(img_arr_)
#     segmented_lungs = segment_lung_region(preprocessed_img)
#     focused_lungs = center_focus_mask(segmented_lungs, preprocessed_img.shape)
#
#     visualize_results(preprocessed_img, segmented_lungs, focused_lungs)


# ----------- secod -----------------
import cv2
import numpy as np
from skimage import measure, morphology, filters, color
import matplotlib.pyplot as plt
import pydicom
from glob import glob

def preprocess_image(img):
    # # Normalize image to the range 0 to 1
    img = img.astype(np.float32) / np.max(img)
    img = cv2.resize(img, (256, 256))
    # Apply Gaussian blur to reduce noise slightly
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img

def segment_lung_region(img):
    # Step 1: Contrast Enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply((img * 255).astype(np.uint8))

    # Step 2: Otsu's Threshold for Binary Segmentation
    thresh_val = filters.threshold_otsu(enhanced_img)
    binary_img = enhanced_img > thresh_val

    return binary_img


def watershed_segmentation10(img):
    # Compute distance transform
    distance_map = cv2.distanceTransform((img * 255).astype(np.uint8), cv2.DIST_L2, 5)
    _, markers = cv2.connectedComponents((img * 255).astype(np.uint8))

    # Apply the Watershed algorithm
    markers = cv2.watershed(np.stack([img] * 3, axis=-1).astype(np.uint8), markers)

    # Convert markers to labels (for visualization)
    return color.label2rgb(markers, bg_label=0)


def watershed_segmentation(img):
    # Step 1: Compute distance transform
    distance_map = cv2.distanceTransform((img * 255).astype(np.uint8), cv2.DIST_L2, 5)

    # Step 2: Find connected components to serve as initial markers
    _, markers = cv2.connectedComponents((img * 255).astype(np.uint8))

    # Step 3: Apply the Watershed algorithm
    markers = cv2.watershed(cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), markers)

    # Step 4: Convert markers to a binary or grayscale image for black-and-white visualization
    output_img = np.zeros_like(img)
    output_img[markers > 1] = 255  # Mark regions with a distinct label as white
    output_img[markers == -1] = 128  # Watershed boundary as gray (optional)

    return output_img


def watershed_segmentation11(img):
    # Ensure input is in the correct format
    img = (img * 255).astype(np.uint8)

    # Step 1: Morphological operations to find sure foreground (nodules) and sure background
    kernel = np.ones((3, 3), np.uint8)

    # Sure background by dilation
    sure_bg = cv2.dilate(img, kernel, iterations=3)

    # Sure foreground by erosion
    sure_fg = cv2.erode(img, kernel, iterations=2)

    # Unknown regions (where the segmentation is uncertain)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Step 2: Marker labelling for watershed
    _, markers = cv2.connectedComponents(sure_fg)

    # Increment markers to distinguish from the background
    markers += 1
    markers[unknown == 255] = 0

    # Step 3: Apply watershed
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color, markers)

    # Convert the result to a label map for visualization
    watershed_img = color.label2rgb(markers, bg_label=0)
    return watershed_img


def detect_edges(segmented_lungs, img_shape):
    # Create a circular mask focusing on the center of the image
    center_x, center_y = img_shape[1] // 2, img_shape[0] // 2
    radius = min(img_shape) // 3

    Y, X = np.ogrid[:img_shape[0], :img_shape[1]]
    dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    mask = dist_from_center <= radius

    return segmented_lungs * mask

def visualize_results(preprocessed_img, segmented_lungs, edges, watershed_img):
    fig, ax = plt.subplots(1, 4, figsize=(24, 6))

    ax[0].imshow(preprocessed_img, cmap='gray')
    ax[0].set_title("Preprocessed Image")

    ax[1].imshow(segmented_lungs, cmap='gray')
    ax[1].set_title("Segmented Lungs")

    ax[2].imshow(edges, cmap='gray')
    ax[2].set_title("Detected Edges (Potential Nodules)")

    ax[3].imshow(watershed_img)
    ax[3].set_title("Watershed Segmentation")

    plt.pause(2)
    plt.close()

# # Example usage with DICOM images
# path = 'Lung_Dx-A0001/04-04-2007-NA-Chest-07990/2.000000-5mm-40805'
# all_imgs = glob(path + "/*.dcm")[20:31]
#
# for file in all_imgs:
#     img_arr = pydicom.dcmread(file)
#     img_arr_ = img_arr.pixel_array
#
#     preprocessed_img = preprocess_image(img_arr_)
#     segmented_lungs = segment_lung_region(preprocessed_img)
#     edges = detect_edges(segmented_lungs, preprocessed_img.shape)
#     watershed_img = watershed_segmentation(segmented_lungs)
#
#     visualize_results(preprocessed_img, segmented_lungs, edges, watershed_img)



# -------------- proposed model ------------
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv3D, MaxPooling3D, Flatten, Dense, Dropout, Bidirectional, GRU, BatchNormalization
)

def build_3d_cnn_bigr_model(input_shape, num_classes, dropout_rate=0.3):
    # Define the input layer for 3D RGB volumes
    input_layer = Input(shape=input_shape)

    # 3D Convolutional Layer 1
    x = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    # 3D Convolutional Layer 2
    x = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    # 3D Convolutional Layer 3
    x = Conv3D(filters=128, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    # Flatten the 3D features
    x = Flatten()(x)

    # Expand dimensions for BiGRU compatibility (if needed)
    x = tf.expand_dims(x, axis=1)

    # BiGRU Layer for temporal/sequence learning
    x = Bidirectional(GRU(64, return_sequences=False))(x)

    # Dropout for regularization
    x = Dropout(dropout_rate)(x)

    # Dense Layer
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_rate)(x)

    # Output Layer
    output_layer = Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    return model


# Parameters
num_samples = 100  # Total samples for training/testing
depth = 10  # Number of slices (previously None)
height, width, channels = 64, 64, 3
num_classes = 2  # Tumor vs No Tumor

# Random data for images
X = np.random.rand(num_samples, depth, height, width, channels).astype(np.float32)

# Random labels (one-hot encoding)
y = np.random.randint(0, num_classes, size=(num_samples,))
y = tf.keras.utils.to_categorical(y, num_classes=num_classes)

# Split into training and testing sets (80% train, 20% test)
split_idx = int(0.8 * num_samples)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Build the model with fixed depth
input_shape = (depth, height, width, channels)  # Fix depth
model = build_3d_cnn_bigr_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=8)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2%}")



# ------------------------------------------------ Granular computing---------------------------------------------------------

# Simulate numerical granule extraction (using 10 rules)
def extract_numerical_granules(sequence):
    """Generates numerical granules from a text sequence using 10 rules."""
    granules = [
        np.mean(sequence),  # Average value of the sequence
        np.max(sequence),   # Maximum value
        np.min(sequence),   # Minimum value
        np.std(sequence),   # Standard deviation
        np.sum(sequence),   # Sum of values
        np.median(sequence),  # Median value
        len(sequence),       # Sequence length
        np.var(sequence),    # Variance
        sequence[-1] if sequence.size > 0 else 0,  # Last element
        sequence[0] if sequence.size > 0 else 0   # First element
    ]
    return np.array(granules)
# for granular features
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample text data
texts = ["This is a sample text.", "Another example for granules."]

# Tokenization
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=300)

# Extract granular features
granular_features = np.array([extract_numerical_granules(seq) for seq in padded_sequences])

print("Granular Features Shape:", granular_features.shape)
# ------------- reverse pancake algorithm -------------
import numpy as np

def reverse_pancake_scramble(sequence):
    """Applies the Reverse Pancake Scramble algorithm to scramble the input sequence."""
    scrambled_sequence = np.array(sequence).copy()
    n = len(scrambled_sequence)
    
    for i in range(n, 0, -1):
        scrambled_sequence[:i] = scrambled_sequence[:i][::-1]
    
    return scrambled_sequence

# Example usage
sequence = [1, 2, 3, 4, 5, 6]
scrambled_sequence = reverse_pancake_scramble(sequence)
print("Original Sequence:", sequence)
print("Scrambled Sequence:", scrambled_sequence.tolist())
""""
# The Reverse Pancake Scramble (RPS) algorithm is inspired by the pancake sorting problem,
where the goal is to sort a sequence by flipping segments. In the RPS context, instead of sorting, 
the goal is to scramble the sequence in a structured way by reversing and flipping segments repeatedly to 
create granular variations of the input.
""""
# ---------------------------- model----------------------------
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, MaxPooling1D, Flatten,
    Bidirectional, LSTM, Dense, Dropout, Concatenate
)

# Simulate Reverse Pancake Scramble Granular (RPSG) algorithm
def reverse_pancake_scramble(sequence):
    """Applies a scrambling operation to generate granular text representation."""
    return np.flip(sequence)

# Simulate numerical granule extraction (using 10 rules)
def extract_numerical_granules(sequence):
    """Generates numerical granules from a text sequence using 10 rules."""
    granules = [
        np.mean(sequence),  # Average value of the sequence
        np.max(sequence),   # Maximum value
        np.min(sequence),   # Minimum value
        np.std(sequence),   # Standard deviation
        np.sum(sequence),   # Sum of values
        np.median(sequence),  # Median value
        len(sequence),       # Sequence length
        np.var(sequence),    # Variance
        sequence[-1] if sequence.size > 0 else 0,  # Last element
        sequence[0] if sequence.size > 0 else 0   # First element
    ]
    return np.array(granules)

# Assume input shape of original sequence
input_shape = (300,)  # Example sequence length
embedding_dim = 300   # Embedding dimension

# Input Layer
input_text = Input(shape=input_shape, name='text_input')

# Embedding Layer
embedding_layer = Embedding(input_dim=20000, output_dim=embedding_dim, input_length=input_shape[0])(input_text)

# Apply RPSG algorithm on input_text (simulated)
rpsg_text = tf.keras.layers.Lambda(lambda x: tf.reverse(x, axis=[-1]))(input_text)

# Extract numerical granules (simulated here as an additional input)
numerical_granules_input = tf.keras.layers.Lambda(lambda x: tf.numpy_function(extract_numerical_granules, [x], tf.float32))(input_text)

# Convolutional Layers (CNN Block)
conv1 = Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(embedding_layer)
pool1 = MaxPooling1D(pool_size=2)(conv1)

# BiLSTM Layer (sequential modeling)
bi_lstm = Bidirectional(LSTM(128, return_sequences=False))(embedding_layer)

# Flatten CNN output
flatten_cnn = Flatten()(pool1)

# Combine the outputs: CNN, BiLSTM, and numerical granules
concat_layer = Concatenate()([flatten_cnn, bi_lstm, numerical_granules_input])

# Fully Connected Layer
dense1 = Dense(128, activation='relu')(concat_layer)
dropout1 = Dropout(0.5)(dense1)

# Output Layer (multi-class classification)
output = Dense(5, activation='softmax')(dropout1)

# Build the Model
model = Model(inputs=input_text, outputs=output)

# Compile the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the Model
model.summary()



# https://www.sciencedirect.com/science/article/abs/pii/S0020025523013312
