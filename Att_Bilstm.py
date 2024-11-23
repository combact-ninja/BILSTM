"""
Lightweight Temporal Attention Encoder module

Credits:
The module is heavily inspired by the works of Vaswani et al. on self-attention and their pytorch implementation of
the Transformer served as code base for the present script.

paper: https://arxiv.org/abs/1706.03762
code: github.com/jadore801120/attention-is-all-you-need-pytorch
"""

import torch
import torch.nn as nn
import numpy as np
import copy
import torch.optim as optim
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.Q = nn.Parameter(torch.zeros((n_head, d_k))).requires_grad_(True)
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def forward(self, q, k, v):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = q.size()

        q = torch.stack([self.Q for _ in range(sz_b)], dim=1).view(-1, d_k)  # (n*b) x d_k

        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk

        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(n_head * sz_b, seq_len, -1)
        output, attn = self.attention(q, k, v)
        attn = attn.view(n_head, sz_b, 1, seq_len)
        attn = attn.squeeze(dim=2)

        output = output.view(n_head, sz_b, 1, d_in // n_head)
        output = output.squeeze(dim=2)

        return output, attn


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2))
        attn = attn / self.temperature

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        return output, attn


def get_sinusoid_encoding_table(positions, d_hid, T=1000):
    ''' Sinusoid position encoding table
    positions: int or list of integer, if int range(positions)'''

    if isinstance(positions, int):
        positions = list(range(positions))

    def cal_angle(position, hid_idx):
        return position / np.power(T, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in positions])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if torch.cuda.is_available():
        return torch.FloatTensor(sinusoid_table).cuda()
    else:
        return torch.FloatTensor(sinusoid_table)


def get_sinusoid_encoding_table_var(positions, d_hid, clip=4, offset=3, T=1000):
    ''' Sinusoid position encoding table
    positions: int or list of integer, if int range(positions)'''

    if isinstance(positions, int):
        positions = list(range(positions))

    x = np.array(positions)

    def cal_angle(position, hid_idx):
        return position / np.power(T, 2 * (hid_idx + offset // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in positions])

    sinusoid_table = np.sin(sinusoid_table)  # dim 2i
    sinusoid_table[:, clip:] = torch.zeros(sinusoid_table[:, clip:].shape)

    if torch.cuda.is_available():
        return torch.FloatTensor(sinusoid_table).cuda()
    else:
        return torch.FloatTensor(sinusoid_table)


class LTAE(nn.Module):
    def __init__(self, len_max_seq, in_channels=128, n_head=16, d_k=8, n_neurons=[256,128], dropout=0.2, d_model=256,
                 T=1000, positions=None, return_att=False):
        """
        Sequence-to-embedding encoder.
        Args:
            in_channels (int): Number of channels of the input embeddings
            n_head (int): Number of attention heads
            d_k (int): Dimension of the key and query vectors
            n_neurons (list): Defines the dimensions of the successive feature spaces of the MLP that processes
                the concatenated outputs of the attention heads
            dropout (float): dropout
            T (int): Period to use for the positional encoding
            len_max_seq (int, optional): Maximum sequence length, used to pre-compute the positional encoding table
            positions (list, optional): List of temporal positions to use instead of position in the sequence
            d_model (int, optional): If specified, the input tensors will first processed by a fully connected layer
                to project them into a feature space of dimension d_model
            return_att (bool): If true, the module returns the attention masks along with the embeddings (default False)

        """

        super(LTAE, self).__init__()
        self.in_channels = in_channels
        self.positions = positions
        self.n_neurons = copy.deepcopy(n_neurons)
        self.return_att = return_att
        self.len_max_seq = len_max_seq

        if positions is None:
            positions = self.len_max_seq + 1

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Sequential(nn.Conv1d(in_channels, d_model, 1),
                                        nn.LayerNorm((d_model, len_max_seq)))
        else:
            self.d_model = in_channels
            self.inconv = None

        sin_tab = get_sinusoid_encoding_table(positions, self.d_model // n_head, T=T)
        self.position_enc = nn.Embedding.from_pretrained(torch.cat([sin_tab for _ in range(n_head)], dim=1),
                                                         freeze=True)

        self.inlayernorm = nn.LayerNorm(self.in_channels)

        self.outlayernorm = nn.LayerNorm(n_neurons[-1])

        self.attention_heads = MultiHeadAttention(
            n_head=n_head, d_k=d_k, d_in=self.d_model)

        assert (self.n_neurons[0] == self.d_model)

        activation = nn.ReLU()

        layers = []
        for i in range(len(self.n_neurons) - 1):
            layers.extend([nn.Linear(self.n_neurons[i], self.n_neurons[i + 1]),
                           nn.BatchNorm1d(self.n_neurons[i + 1]),
                           activation])

        self.mlp = nn.Sequential(*layers)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        sz_b, seq_len, d = x.shape
        self.len_max_seq = x.shape[1]
        x = self.inlayernorm(x)

        if self.inconv is not None:
            x = self.inconv(x.permute(0, 2, 1)).permute(0, 2, 1)

        if self.positions is None:
            src_pos = torch.arange(1, seq_len + 1, dtype=torch.long).expand(sz_b, seq_len).to(x.device)
        else:
            src_pos = torch.arange(0, seq_len, dtype=torch.long).expand(sz_b, seq_len).to(x.device)
        enc_output = x + self.position_enc(src_pos)

        enc_output, attn = self.attention_heads(enc_output, enc_output, enc_output)

        enc_output = enc_output.permute(1, 0, 2).contiguous().view(sz_b, -1)  # Concatenate heads

        enc_output = self.outlayernorm(self.dropout(self.mlp(enc_output)))

        if self.return_att:
            return enc_output, attn
        else:
            return enc_output



# ---------------------- torch Bilstm module --------------------
import torch
import torch.nn as nn
import torch.optim as optim

# Define the BiLSTM model
class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BiLSTMModel, self).__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(2 * hidden_dim, 16)  # Fully connected layer (hidden_dim doubled for BiLSTM)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, output_dim)  # Output layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.bilstm(x)# LSTM outputs (output, (h_n, c_n))
        att = LTAE(len_max_seq=lstm_out.shape[1])(lstm_out)
        # lstm_out = lstm_out[:, -1, :]  # Take the output of the last time step
        dense_out = self.relu(self.fc1(att))  # Fully connected layer with ReLU
        output = self.sigmoid(self.fc2(dense_out))  # Final output with Sigmoid
        return output

# Set up input data and labels
input_data = torch.rand(1000, 66, 1)  # Shape: (batch_size, sequence_length, input_dim)
labels = torch.randint(0, 2, (1000, 1)).float()  # Binary labels

# Hyperparameters
input_dim = 1
hidden_dim = 64
output_dim = 1
batch_size = 32
num_epochs = 10

# Instantiate the model, define loss and optimizer
model = BiLSTMModel(input_dim, hidden_dim, output_dim)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
dataset = torch.utils.data.TensorDataset(input_data, labels)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_data, batch_labels in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}")









