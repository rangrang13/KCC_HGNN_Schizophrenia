
# train.py

import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

from config import (
    SEED,
    DEVICE,
    BATCH_SIZE,
    GRAD_ACCUM_STEPS,
    LR,
    WEIGHT_DECAY,
    EPOCHS,
    NUM_FOLDS,
    CHECKPOINT_DIR,
    USE_EARLY_STOPPING,
    PATIENCE,
    SAVE_BEST_MODEL,
    NODE_FEATURE_MODE,
    INPUT_DIM,
    USE_THRESHOLD_TUNING,
    BASE_THRESHOLD,
    RECALL_TOLERANCE,
    THRESHOLD_SEARCH_START,
    THRESHOLD_SEARCH_END,
    THRESHOLD_SEARCH_STEP,
)

from dataset import COBREHypergraphDataset, simple_collate_fn
from model import SubjectHGNNClassifier
from utils import (
    set_seed,
    AverageMeter,
    compute_classification_metrics,
    compute_metrics,
    find_best_threshold_under_recall_constraint,
    save_checkpoint,
    load_checkpoint,
    EarlyStopping,
    print_epoch_log,
    print_fold_summary,
)


# 학습 안정화 설정
GRAD_CLIP_NORM = 1.0
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_MIN_LR = 1e-4


# 데이터 및 하이퍼그래프 이동
def move_sample_to_device(sample, device):
    x = sample["x"].to(device)
    y = sample["y"].to(device).view(1)
    hg = sample["hg"].to(device)
    return x, y, hg


# 클래스 가중치 계산
def get_train_class_weights(dataset, train_idx, device):
    all_labels = dataset.get_labels()
    train_labels = [all_labels[i] for i in train_idx]

    class_counts = np.bincount(train_labels, minlength=2).astype(np.float32)
    class_counts[class_counts == 0] = 1.0

    total = class_counts.sum()
    class_weights = total / (2.0 * class_counts)

    return torch.tensor(class_weights, dtype=torch.float32, device=device)


# 에포크별 모델 학습
def train_one_epoch(model, loader, criterion, optimizer, device, grad_accum_steps=1):
    model.train()

    loss_meter = AverageMeter()
    y_true_all = []
    y_pred_all = []
    y_prob_all = []

    optimizer.zero_grad()

    for step, sample in enumerate(loader, start=1):
        x, y, hg = move_sample_to_device(sample, device)

        logits, _, _ = model(x, hg)
        loss = criterion(logits, y)

        scaled_loss = loss / grad_accum_steps
        scaled_loss.backward()

        if (step % grad_accum_steps == 0) or (step == len(loader)):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=GRAD_CLIP_NORM,
            )
            optimizer.step()
            optimizer.zero_grad()

        loss_meter.update(loss.item(), n=1)

        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)

        y_true_all.append(y.item())
        y_pred_all.append(pred.item())
        y_prob_all.append(probs[:, 1].item())

    metrics = compute_classification_metrics(
        y_true_all, y_pred_all, y_prob_all
    )
    return loss_meter.avg, metrics


# 검증 데이터 평가
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    loss_meter = AverageMeter()
    y_true_all = []
    y_pred_all = []
    y_prob_all = []

    for sample in loader:
        x, y, hg = move_sample_to_device(sample, device)

        logits, _, _ = model(x, hg)
        loss = criterion(logits, y)

        loss_meter.update(loss.item(), n=1)

        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)

        y_true_all.append(y.item())
        y_pred_all.append(pred.item())
        y_prob_all.append(probs[:, 1].item())

    metrics = compute_classification_metrics(
        y_true_all, y_pred_all, y_prob_all
    )
    return loss_meter.avg, metrics


# 피험자별 예측 확률 수집
@torch.no_grad()
def collect_probs(model, dataset, indices, device=DEVICE):
    model.eval()

    y_true_all = []
    y_prob_all = []

    for idx in indices:
        sample = dataset[idx]
        x, y, hg = move_sample_to_device(sample, device)

        logits, _, _ = model(x, hg)
        probs = torch.softmax(logits, dim=1)

        y_true_all.append(y.item())
        y_prob_all.append(probs[:, 1].item())

    return np.array(y_true_all), np.array(y_prob_all)


# 데이터 로더 구성
def build_dataloaders(dataset, train_idx, val_idx):
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=simple_collate_fn,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=simple_collate_fn,
    )

    return train_loader, val_loader


# 폴드별 학습 및 평가
def run_fold(fold, dataset, train_idx, val_idx):
    print("=" * 60)
    print(f"Start Fold {fold}")
    print("=" * 60)

    train_loader, val_loader = build_dataloaders(
        dataset, train_idx, val_idx
    )

    # 모델 및 옵티마이저
    model = SubjectHGNNClassifier().to(DEVICE)

    class_weights = get_train_class_weights(dataset, train_idx, DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # 학습률 스케줄러
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
        min_lr=LR_SCHEDULER_MIN_LR,
    )

    print(f"[Fold {fold}] Class weights: {class_weights.detach().cpu().numpy()}")

    # 조기 종료
    early_stopper = (
        EarlyStopping(patience=PATIENCE, mode="min")
        if USE_EARLY_STOPPING else None
    )

    best_val_loss = np.inf
    best_ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_fold_{fold}.pt")

    # 모델 학습
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=DEVICE,
            grad_accum_steps=GRAD_ACCUM_STEPS,
        )

        val_loss, val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=DEVICE,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print_epoch_log(
            fold=fold,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            lr=current_lr,
        )

        # 최적 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            if SAVE_BEST_MODEL:
                save_checkpoint(
                    save_path=best_ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_score=best_val_loss,
                    extra={
                        "fold": int(fold),
                        "best_metric_name": "val_loss",
                        "train_idx": train_idx.tolist(),
                        "val_idx": val_idx.tolist(),
                        "node_feature_mode": NODE_FEATURE_MODE,
                        "input_dim": INPUT_DIM,
                    },
                )

        scheduler.step(val_loss)

        if early_stopper is not None:
            if early_stopper.step(val_loss):
                print(f"[Fold {fold}] Early stopping at epoch {epoch}")
                break

    # 최적 모델 복원
    if SAVE_BEST_MODEL and os.path.exists(best_ckpt_path):
        load_checkpoint(
            best_ckpt_path,
            model=model,
            optimizer=None,
            map_location=DEVICE,
        )

    final_val_loss, _ = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=DEVICE,
    )

    # Train fold 기반 임계값 탐색
    if USE_THRESHOLD_TUNING:
        y_true_train, y_prob_train = collect_probs(
            model, dataset, train_idx, device=DEVICE
        )

        best_thr, _, _ = find_best_threshold_under_recall_constraint(
            y_true=y_true_train,
            y_prob=y_prob_train,
            base_threshold=BASE_THRESHOLD,
            recall_tolerance=RECALL_TOLERANCE,
            search_start=THRESHOLD_SEARCH_START,
            search_end=THRESHOLD_SEARCH_END,
            search_step=THRESHOLD_SEARCH_STEP,
        )
    else:
        best_thr = BASE_THRESHOLD

    # 최종 검증 성능
    y_true_val, y_prob_val = collect_probs(
        model, dataset, val_idx, device=DEVICE
    )

    final_val_metrics = compute_metrics(
        y_true=y_true_val,
        y_prob=y_prob_val,
        threshold=best_thr,
    )

    final_val_metrics["loss"] = float(final_val_loss)

    print(f"[Fold {fold}] Best Val Loss: {best_val_loss:.5f}")
    print_fold_summary(
        fold=fold,
        metrics=final_val_metrics,
    )

    return final_val_metrics


# 교차검증 결과 요약
def summarize_cross_validation_results(fold_results):
    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "sensitivity",
        "specificity",
        "auc",
    ]

    print("\n" + "=" * 60)
    print("Cross-Validation Final Summary")
    print("=" * 60)

    for metric_name in metric_names:
        values = [result[metric_name] for result in fold_results]
        mean_val = np.nanmean(values)
        std_val = np.nanstd(values)

        print(f"{metric_name:12s}: {mean_val:.5f} ± {std_val:.5f}")

    threshold_values = [result["threshold"] for result in fold_results]
    print(
        f"{'threshold':12s}: "
        f"{np.nanmean(threshold_values):.5f} ± "
        f"{np.nanstd(threshold_values):.5f}"
    )

    loss_values = [result["loss"] for result in fold_results]
    print(
        f"{'loss':12s}: "
        f"{np.nanmean(loss_values):.5f} ± "
        f"{np.nanstd(loss_values):.5f}"
    )

    print("=" * 60)


# 전체 학습 실행
def main():
    set_seed(SEED)

    print("=" * 60)
    print("Train HGNN for Subject-level Hypergraph Classification")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Node feature mode: {NODE_FEATURE_MODE}")
    print(f"Input dim: {INPUT_DIM}")

    dataset = COBREHypergraphDataset()
    labels = dataset.get_labels()

    print(f"Total usable subjects: {len(dataset)}")

    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")

    # Stratified 5-fold 교차검증
    skf = StratifiedKFold(
        n_splits=NUM_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(labels)), labels), start=1
    ):
        fold_metrics = run_fold(
            fold=fold,
            dataset=dataset,
            train_idx=train_idx,
            val_idx=val_idx,
        )
        fold_results.append(fold_metrics)

    summarize_cross_validation_results(fold_results)


if __name__ == "__main__":
    main()
