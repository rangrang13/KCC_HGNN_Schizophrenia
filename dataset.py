# dataset.py

from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from dhg import Hypergraph

from config import (
    CONTROL_DIR,
    PATIENT_DIR,
    EXCLUDE_FILES,
    LABEL_MAP,
    NUM_ROIS,
    USE_ABS_CORR,
    K_NEIGHBORS,
    INCLUDE_SELF_IN_HYPEREDGE,
    NUM_HYPEREDGES_PER_SUBJECT,
    CORR_ROW_USE_ABS,
    CORR_ROW_ZERO_DIAG,
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

def unique_hyperedges(hyperedges: List[List[int]]) -> List[List[int]]:
    seen = set()
    unique_edges = []
    for edge in hyperedges:
        edge_sorted = tuple(sorted(set(edge)))
        if len(edge_sorted) == 0:
            continue
        if edge_sorted not in seen:
            seen.add(edge_sorted)
            unique_edges.append(list(edge_sorted))
    return unique_edges

def limit_num_hyperedges(hyperedges: List[List[int]], max_edges: Optional[int]) -> List[List[int]]:
    if max_edges is None:
        return hyperedges
    if max_edges <= 0:
        raise ValueError("NUM_HYPEREDGES_PER_SUBJECT는 양수여야 한다.")
    return hyperedges[:max_edges]

def build_hyperedges_from_corr(
    corr: np.ndarray,
    k_neighbors: int = K_NEIGHBORS,
    use_abs_corr: bool = USE_ABS_CORR,
    include_self: bool = INCLUDE_SELF_IN_HYPEREDGE,
    max_edges: Optional[int] = NUM_HYPEREDGES_PER_SUBJECT,
) -> List[List[int]]:
    num_rois = corr.shape[0]
    hyperedges = []
    score_matrix = np.abs(corr) if use_abs_corr else corr.copy()
    for anchor in range(num_rois):
        scores = score_matrix[anchor].copy()
        scores[anchor] = -np.inf
        topk_indices = np.argsort(scores)[-k_neighbors:]
        topk_indices = topk_indices[np.argsort(scores[topk_indices])[::-1]]
        if include_self:
            hyperedge = [anchor] + topk_indices.tolist()
        else:
            hyperedge = topk_indices.tolist()
        hyperedges.append(hyperedge)
    hyperedges = unique_hyperedges(hyperedges)
    hyperedges = limit_num_hyperedges(hyperedges, max_edges)
    return hyperedges

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


def load_single_subject(file_path: Path, folder_group_name: str) -> Dict[str, Any]:
    with np.load(file_path, allow_pickle=False) as data:
        corr_data = data["corr"]
    corr_data = symmetrize_corr(corr_data)
    group_name = folder_group_name
    x = build_corr_row_node_features(corr_data)
    hyperedges = build_hyperedges_from_corr(corr_data)
    hg = Hypergraph(NUM_ROIS, hyperedges)
    y = LABEL_MAP[group_name]
    sample = {
        "x": torch.tensor(x, dtype=torch.float32),
        "hg": hg,
        "y": torch.tensor(y, dtype=torch.long),
        "file_name": file_path.name,
        "subject_id": file_path.stem,
        "group_name": group_name,
        "num_hyperedges": len(hyperedges),
    }
    return sample


class COBREHypergraphDataset(Dataset):
    def __init__(self, cache_in_memory: bool = True):
        super().__init__()
        self.samples = collect_subject_files()
        self.cache_in_memory = cache_in_memory
        self._cache: Dict[int, Dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
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
            group_name = item["group_name"]
            labels.append(LABEL_MAP[group_name])
        return labels

def simple_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(batch) != 1:
        raise ValueError("Batch size는 1이어야 한다.")
    return batch[0]