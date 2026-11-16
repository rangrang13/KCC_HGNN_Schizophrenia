# dataset_gcn.py

from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from config_gcn import (
    CONTROL_DIR,
    PATIENT_DIR,
    EXCLUDE_FILES,
    LABEL_MAP,
    CORR_ROW_USE_ABS,
    CORR_ROW_ZERO_DIAG,
    USE_ABS_CORR_FOR_GRAPH,
    K_NEIGHBORS,
    INCLUDE_SELF_LOOP,
    MAKE_UNDIRECTED,
)


def collect_subject_files() -> List[Dict[str, Any]]:
    samples = []

    for group_name, group_dir in [("Control", CONTROL_DIR), ("Patient", PATIENT_DIR)]:
        group_path = Path(group_dir)

        for file_path in sorted(group_path.rglob("*.npz")):
            if file_path.name in EXCLUDE_FILES:
                continue

            samples.append({
                "file_path": file_path,
                "group_name": group_name,
            })

    return samples


def symmetrize_corr(corr: np.ndarray) -> np.ndarray:
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    return corr.astype(np.float32)


def rowwise_zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    row_mean = x.mean(axis=1, keepdims=True)
    row_std = x.std(axis=1, keepdims=True)
    row_std = np.where(row_std < eps, 1.0, row_std)
    return (x - row_mean) / row_std


def build_corr_row_node_features(corr_data: np.ndarray) -> np.ndarray:
    x = corr_data.copy()

    if CORR_ROW_USE_ABS:
        x = np.abs(x)

    if CORR_ROW_ZERO_DIAG:
        np.fill_diagonal(x, 0.0)

    x = rowwise_zscore(x)

    if CORR_ROW_ZERO_DIAG:
        np.fill_diagonal(x, 0.0)

    return x.astype(np.float32)


def build_edge_index_from_corr(corr: np.ndarray) -> torch.Tensor:
    num_rois = corr.shape[0]
    score_matrix = np.abs(corr) if USE_ABS_CORR_FOR_GRAPH else corr.copy()

    edge_pairs = []

    for src in range(num_rois):
        scores = score_matrix[src].copy()
        scores[src] = -np.inf

        topk_indices = np.argsort(scores)[-K_NEIGHBORS:]
        topk_indices = topk_indices[np.argsort(scores[topk_indices])[::-1]]

        for dst in topk_indices.tolist():
            edge_pairs.append((src, dst))

    if MAKE_UNDIRECTED:
        extra_pairs = []
        for src, dst in edge_pairs:
            extra_pairs.append((dst, src))
        edge_pairs.extend(extra_pairs)

    if INCLUDE_SELF_LOOP:
        for i in range(num_rois):
            edge_pairs.append((i, i))

    edge_pairs = list(dict.fromkeys(edge_pairs))
    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()

    return edge_index


def load_single_subject(file_path: Path, folder_group_name: str) -> Data:
    with np.load(file_path, allow_pickle=False) as data:
        corr_data = data["corr"]

    corr_data = symmetrize_corr(corr_data)

    x = build_corr_row_node_features(corr_data)
    edge_index = build_edge_index_from_corr(corr_data)
    y = LABEL_MAP[folder_group_name]

    data_obj = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor([y], dtype=torch.long),
    )

    data_obj.file_name = file_path.name
    data_obj.subject_id = file_path.stem
    data_obj.group_name = folder_group_name

    return data_obj


class COBREGraphPYGDataset(Dataset):
    def __init__(self, cache_in_memory: bool = True):
        super().__init__()
        self.samples = collect_subject_files()
        self.cache_in_memory = cache_in_memory
        self._cache: Dict[int, Data] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Data:
        if self.cache_in_memory and idx in self._cache:
            return self._cache[idx]

        item_info = self.samples[idx]
        file_path = item_info["file_path"]
        group_name = item_info["group_name"]

        sample = load_single_subject(file_path, group_name)

        if self.cache_in_memory:
            self._cache[idx] = sample

        return sample

    def get_labels(self) -> List[int]:
        labels = []
        for item in self.samples:
            labels.append(LABEL_MAP[item["group_name"]])
        return labels


def simple_collate_fn(batch: List[Data]) -> Data:
    if len(batch) != 1:
        raise ValueError("Batch size는 1이어야한다.")
    return batch[0]