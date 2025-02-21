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


# ----------------- ganbert -----
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Simulate data (replace with your real data)
np.random.seed(42)
classes = 7
total_samples = 1000
class_counts = [300, 200, 150, 100, 80, 50, 20]  # Imbalanced
labeled_samples = sum(class_counts)
unlabeled_samples = total_samples - labeled_samples

texts = []
labels = []
for i, count in enumerate(class_counts):
    texts.extend([f"Sample text for class {i} number {j}" for j in range(count)])
    labels.extend([i] * count)
texts.extend([f"Unlabeled sample {i}" for i in range(unlabeled_samples)])
labels.extend([-1] * unlabeled_samples)

df = pd.DataFrame({"text": texts, "labels": labels})
labeled_df = df[df["labels"] != -1].sample(frac=0.8, random_state=42)
unlabeled_df = df[df["labels"] == -1]
test_df = df[df["labels"] != -1].drop(labeled_df.index)

# Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_len, return_token_type_ids=False,
            padding="max_length", truncation=True, return_attention_mask=True, return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# Load data
train_texts = labeled_df["text"].tolist() + unlabeled_df["text"].tolist()
train_labels = labeled_df["labels"].tolist() + unlabeled_df["labels"].tolist()
test_texts = test_df["text"].tolist()
test_labels = test_df["labels"].tolist()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_dataset = TextDataset(train_texts, train_labels, tokenizer)
test_dataset = TextDataset(test_texts, test_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Models
class Generator(torch.nn.Module):
    def __init__(self, bert_model):
        super(Generator, self).__init__()
        self.bert = bert_model

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

class Discriminator(torch.nn.Module):
    def __init__(self, num_classes=7):
        super(Discriminator, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_classes + 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

# Initialize models and optimizers
generator = Generator(BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=8)).to(device)
discriminator = Discriminator(num_classes=7).to(device)
gen_optimizer = torch.optim.AdamW(generator.parameters(), lr=2e-5)
disc_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=2e-5)
ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

# Training function
def train_ganbert(generator, discriminator, train_loader, gen_optimizer, disc_optimizer, epochs=3):
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        total_disc_loss = 0
        total_gen_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            disc_optimizer.zero_grad()
            with torch.no_grad():
                gen_outputs = generator(input_ids, attention_mask)
            disc_outputs = discriminator(input_ids, attention_mask)
            labeled_mask = labels != -1
            unlabeled_mask = labels == -1

            disc_sup_loss = ce_loss(disc_outputs[labeled_mask], labels[labeled_mask]) if labeled_mask.sum() > 0 else 0
            real_labels = torch.ones(len(labels), dtype=torch.long).to(device) * 7
            real_labels[labeled_mask] = labels[labeled_mask]
            disc_adv_loss = ce_loss(disc_outputs, real_labels)
            disc_loss = disc_sup_loss + disc_adv_loss
            disc_loss.backward()
            disc_optimizer.step()
            total_disc_loss += disc_loss.item()

            gen_optimizer.zero_grad()
            gen_outputs = generator(input_ids, attention_mask)
            disc_outputs = discriminator(input_ids, attention_mask)
            gen_loss = ce_loss(disc_outputs, labels.clamp(min=0))
            gen_loss.backward()
            gen_optimizer.step()
            total_gen_loss += gen_loss.item()

        print(f"Epoch {epoch + 1}: Disc Loss = {total_disc_loss / len(train_loader):.4f}, Gen Loss = {total_gen_loss / len(train_loader):.4f}")

# Extract balanced data
def extract_balanced_data(discriminator, train_texts, train_labels, num_classes=7, samples_per_class=143):
    discriminator.eval()
    all_preds = []
    dataset = TextDataset(train_texts, train_labels, tokenizer)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting labels"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = discriminator(input_ids, attention_mask)
            preds = torch.argmax(outputs[:, :7], dim=1)
            all_preds.extend(preds.cpu().numpy())

    pred_df = pd.DataFrame({"text": train_texts, "predicted_label": all_preds})
    balanced_df = pd.DataFrame()
    for cls in range(num_classes):
        class_samples = pred_df[pred_df["predicted_label"] == cls]
        if len(class_samples) > samples_per_class:
            balanced_samples = class_samples.sample(n=samples_per_class, random_state=42)
        else:
            balanced_samples = class_samples.sample(n=samples_per_class, replace=True, random_state=42)
        balanced_df = pd.concat([balanced_df, balanced_samples])

    return balanced_df

# Evaluation function
def evaluate(discriminator, test_loader):
    discriminator.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = discriminator(input_ids, attention_mask)
            preds = torch.argmax(outputs[:, :7], dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=[f"Class {i}" for i in range(7)]))

# Run everything
train_ganbert(generator, discriminator, train_loader, gen_optimizer, disc_optimizer, epochs=3)
balanced_df = extract_balanced_data(discriminator, train_texts, train_labels, num_classes=7, samples_per_class=143)
balanced_df.to_csv("balanced_data.csv", index=False)
print("Balanced data distribution:")
print(balanced_df["predicted_label"].value_counts())
evaluate(discriminator, test_loader)


# Load your real data
df = pd.read_csv("your_data.csv")  # Should have "text" and "labels" (0-6 or -1 for unlabeled)
labeled_df = df[df["labels"] != -1].sample(frac=0.8, random_state=42)
unlabeled_df = df[df["labels"] == -1]
test_df = df[df["labels"] != -1].drop(labeled_df.index)


