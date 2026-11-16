# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    INPUT_DIM,
    HIDDEN_DIM,
    NUM_CLASSES,
    DROPOUT,
    NUM_LAYERS,
    POOLING,
)


# HGNN 합성곱 블록
class HGNNConvBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.theta = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    # 가중치 초기화
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.theta.weight)
        if self.theta.bias is not None:
            nn.init.zeros_(self.theta.bias)

    def forward(self, x: torch.Tensor, hg):
        x = self.theta(x)
        x = hg.smoothing_with_HGNN(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


# 피험자 단위 HGNN 분류 모델
class SubjectHGNNClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT,
        pooling: str = POOLING,
    ):
        super().__init__()
        # 풀링 방식 확인
        if pooling not in ("mean", "max"):
            raise ValueError(f"지원하지 않는 풀링 : {pooling}")
        self.pooling = pooling
        # 입력 특징 변환
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        # HGNN 레이어
        self.convs = nn.ModuleList()
        for _ in range(NUM_LAYERS):
            self.convs.append(
                HGNNConvBlock(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    dropout=dropout,
                )
            )
        # 최종 분류기
        self.pre_classifier_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.reset_parameters()

    # 가중치 초기화
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor, hg):
        # 입력 특징 변환
        x = self.input_proj(x)
        x = F.relu(x)
        x = self.input_dropout(x)
        # 하이퍼그래프 메시지 전달
        for conv in self.convs:
            x = conv(x, hg)
        node_emb = x
        # 평균 / 최대 풀링
        if self.pooling == "mean":
            subject_emb = node_emb.mean(dim=0, keepdim=True)
        else:
            subject_emb = node_emb.max(dim=0, keepdim=True).values
        # 질병 분류
        subject_emb = self.pre_classifier_dropout(subject_emb)
        logits = self.classifier(subject_emb)
        return logits, subject_emb, node_emb