# model_gcn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from config_gcn import (
    INPUT_DIM,
    HIDDEN_DIM,
    NUM_CLASSES,
    DROPOUT,
    NUM_LAYERS,
)


class GCNConvBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()

        self.conv = GCNConv(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = self.conv(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class SubjectGCNClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList()
        for _ in range(NUM_LAYERS):
            self.convs.append(
                GCNConvBlock(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    dropout=dropout,
                )
            )

        self.pre_classifier_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)

        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index

        x = self.input_proj(x)
        x = F.relu(x)
        x = self.input_dropout(x)

        for conv in self.convs:
            x = conv(x, edge_index)

        node_emb = x
        subject_emb = node_emb.mean(dim=0, keepdim=True)
        subject_emb = self.pre_classifier_dropout(subject_emb)
        logits = self.classifier(subject_emb)

        return logits, subject_emb, node_emb
