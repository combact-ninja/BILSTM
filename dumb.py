import pandas as pd
import random
from datetime import datetime, timedelta

class VMScheduler:
    def __init__(self, blocklist_table, resource_capacity, vm_list):
        # Blocklist table (e.g., blocked VM IDs or IPs)
        self.blocklist = blocklist_table
        # Resource capacity (e.g., CPU, memory limits)
        self.resource_capacity = resource_capacity
        # List of virtual machines with IDs, resource demands
        self.vm_list = vm_list

    def is_blocked(self, vm_id):
        # Check if the VM ID is in the blocklist
        return vm_id in self.blocklist

    def schedule_vms(self):
        # Schedule VMs based on available resources and blocklist
        scheduled_vms = []
        available_capacity = self.resource_capacity.copy()
        
        for vm in self.vm_list:
            if self.is_blocked(vm['vm_id']):
                print(f"VM {vm['vm_id']} is blocked. Skipping...")
                continue

            # Check if resources are available for the VM
            if available_capacity['CPU'] >= vm['cpu'] and available_capacity['Memory'] >= vm['memory']:
                # Schedule the VM and allocate resources
                scheduled_vms.append(vm)
                available_capacity['CPU'] -= vm['cpu']
                available_capacity['Memory'] -= vm['memory']
                print(f"Scheduled VM {vm['vm_id']} with CPU: {vm['cpu']} and Memory: {vm['memory']}")
            else:
                print(f"Not enough resources for VM {vm['vm_id']}")

        return scheduled_vms

# Sample Blocklist and VM List
blocklist_table = ['VM102', 'VM105']
resource_capacity = {'CPU': 16, 'Memory': 64}  # Example: 16 CPU cores, 64GB memory
vm_list = [
    {'vm_id': 'VM101', 'cpu': 4, 'memory': 16},
    {'vm_id': 'VM102', 'cpu': 6, 'memory': 24},  # Blocked VM
    {'vm_id': 'VM103', 'cpu': 8, 'memory': 32},
    {'vm_id': 'VM104', 'cpu': 3, 'memory': 12},
    {'vm_id': 'VM105', 'cpu': 2, 'memory': 8}    # Blocked VM
]

# Create Scheduler Instance
scheduler = VMScheduler(blocklist_table, resource_capacity, vm_list)

# Schedule VMs
scheduled_vms = scheduler.schedule_vms()
print(f"\nScheduled VMs: {scheduled_vms}")


import tensorflow as tf
from tensorflow.keras.layers import Layer, LSTM, Bidirectional, Dropout, Dense, LeakyReLU, Softmax
import numpy as np

class GraphAttentionLayer(Layer):
    def __init__(self, in_features, out_features, n_heads, dropout=0.4, leaky_relu_slope=0.2):
        super(GraphAttentionLayer, self).__init__()

        self.n_heads = n_heads
        self.out_features = out_features
        self.dropout = dropout

        # Initialize weights
        self.W = self.add_weight(
            shape=(in_features, out_features * n_heads), initializer='glorot_uniform', trainable=True
        )
        self.a = self.add_weight(
            shape=(n_heads, 2 * out_features, 1), initializer='glorot_uniform', trainable=True
        )

        self.leaky_relu = LeakyReLU(alpha=leaky_relu_slope)
        self.dropout_layer = Dropout(dropout)
        self.softmax = Softmax(axis=-1)

    def call(self, h, adj_mat, training=False):
        # Apply transformation
        h_transformed = tf.matmul(h, self.W)
        h_transformed = self.dropout_layer(h_transformed, training=training)
        
        # Reshape for multi-head attention
        h_transformed = tf.reshape(h_transformed, (-1, self.n_heads, h.shape[1], self.out_features))
        
        # Compute attention scores
        source_scores = tf.matmul(h_transformed, self.a[:, :self.out_features, :])
        target_scores = tf.matmul(h_transformed, self.a[:, self.out_features:, :])
        e = source_scores + tf.transpose(target_scores, perm=[0, 1, 3, 2])
        e = self.leaky_relu(e)

        # Apply mask and attention
        e = tf.where(adj_mat > 0, e, tf.fill(tf.shape(e), -1e9))
        attention = self.softmax(e)
        attention = self.dropout_layer(attention, training=training)
        
        # Compute output
        h_prime = tf.matmul(attention, h_transformed)
        return h_prime

class DigitalTwinBiLSTM_GAT(tf.keras.Model):
    def __init__(self, in_features, n_hidden, n_heads, num_classes):
        super(DigitalTwinBiLSTM_GAT, self).__init__()
        self.bilstm = Bidirectional(LSTM(n_hidden, return_sequences=True))
        self.gat = GraphAttentionLayer(n_hidden, n_hidden, n_heads)
        self.dense = Dense(num_classes, activation='softmax')

    def call(self, input_tensor, adj_mat, training=False):
        x = self.bilstm(input_tensor)
        x = self.gat(x, adj_mat, training=training)
        return self.dense(x)

# Example Usage
# input_tensor: Shape (batch_size, sequence_length, input_features)
# adj_mat: Adjacency matrix for graph attention

model = DigitalTwinBiLSTM_GAT(in_features=16, n_hidden=32, n_heads=4, num_classes=2)


# ===========================================================================================
import tensorflow as tf
from tensorflow.keras.layers import Layer, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras import Model

# Sample BiLSTM Model for Anomaly Detection
class BiLSTMAnomalyDetectionModel(Model):
    def __init__(self, input_shape):
        super(BiLSTMAnomalyDetectionModel, self).__init__()
        self.lstm = Bidirectional(LSTM(64, return_sequences=True, dropout=0.4))
        self.dense = Dense(1, activation='sigmoid')  # Output (0: Normal, 1: Anomaly)

    def call(self, inputs, training=False):
        x = self.lstm(inputs)
        x = self.dense(x)
        return x

# VM Scheduler with Blocklist Logic
class VMScheduler:
    def __init__(self):
        self.blocklist = set()  # Store blocked VMs
        self.scheduled_vms = set()  # Store scheduled VMs

    def schedule_vm(self, vm_id, prediction):
        """
        If an anomaly is detected (prediction = 1), block the VM from scheduling.
        If no anomaly (prediction = 0), schedule the VM.
        """
        if prediction == 1:
            # Block the VM from scheduling
            self.blocklist.add(vm_id)
            print(f"VM {vm_id} is blocked due to anomaly prediction.")
        else:
            # If no anomaly detected, schedule the VM
            if vm_id not in self.blocklist:
                self.scheduled_vms.add(vm_id)
                print(f"VM {vm_id} has been successfully scheduled.")
            else:
                print(f"VM {vm_id} is still blocked and cannot be scheduled.")

# Data Preprocessing for the BiLSTM Model
def preprocess_data(data):
    """
    Preprocess the data to a format suitable for the BiLSTM model.
    """
    # Add your data preprocessing steps here (e.g., normalization, padding)
    return data

# Example Function for Running the VM Scheduler
def run_vm_scheduler(model, scheduler, data, vm_id):
    """
    Runs the prediction and scheduler logic.
    """
    # Step 1: Preprocess the incoming data
    processed_data = preprocess_data(data)
    
    # Step 2: Predict if there's an anomaly using the BiLSTM model
    prediction = model(processed_data)  # Shape: (batch_size, n_nodes, 1)
    
    # Step 3: Convert model output (Sigmoid) to a binary prediction
    predicted_label = tf.round(prediction[-1, -1, 0]).numpy()  # Get the last prediction (e.g., for the last timestep)
    
    # Step 4: Make scheduling decision based on anomaly prediction
    scheduler.schedule_vm(vm_id, predicted_label)

# Create a sample model instance
model = BiLSTMAnomalyDetectionModel(input_shape=(None, 290, 16))  # Modify shape according to your input

# Create the VM Scheduler instance
scheduler = VMScheduler()

# Example: Simulate some data for VM with id 'vm_1'
# Assuming data is in the form (batch_size, n_nodes, features), e.g., (16, 290, 16)
data_example = tf.random.normal([16, 290, 16])

# Run the VM Scheduler with prediction
run_vm_scheduler(model, scheduler, data_example, vm_id="vm_1")
