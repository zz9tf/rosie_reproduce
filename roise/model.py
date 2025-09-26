"""
Model and training orchestration module for H&E -> multiplex protein prediction.

This module encapsulates the model architecture, dataset wiring, training loop,
and evaluation utilities into a single cohesive, importable unit. It provides a
`ProteinPredictor` class with ergonomic methods for training and evaluation.
"""

from __future__ import annotations

import os
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from torchvision import models

# Local imports
from patch_dataset import PatchImageDataset, get_default_transforms, collate_fn


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_sample_data_format() -> pd.DataFrame:
    """创建示例数据格式说明。"""
    sample_data = {
        "image_path": ["patient1/slide1.png", "patient1/slide2.png", "patient2/slide1.png"],
        "image_id": ["img_001", "img_002", "img_003"],
        "CD3": [0.5, 0.8, 0.3],
        "CD8": [0.2, 0.6, 0.4],
        "CD20": [0.7, 0.3, 0.9],
    }
    return pd.DataFrame(sample_data)


def ensure_data_file(data_file: str) -> None:
    """确保数据 parquet 存在；若缺失则写入一个模板。"""
    if os.path.exists(data_file):
        return
    logger.warning(f"❌ 数据文件不存在: {data_file}")
    os.makedirs(os.path.dirname(data_file), exist_ok=True)
    sample_df = create_sample_data_format()
    sample_df.to_parquet(data_file)
    logger.info(f"✅ 示例数据已保存到: {data_file}")


def build_datasets_and_loaders(
    *,
    root_dir: str,
    image_dir: Optional[str] = None,
    data_file: Optional[str] = None,
    patch_size: int = 128,
    target_biomarkers: Optional[List[str]] = None,
    dataset_splits: Optional[Dict[str, List[str]]] = None,
    num_workers: int = 16,
    batch_size: int = 256,
    use_zarr: bool = True,
    zarr_marker: str = 'HE',
) -> Tuple[DataLoader, DataLoader, PatchImageDataset, PatchImageDataset]:
    """构建数据集与 DataLoader。

    Returns:
        (train_loader, val_loader, train_dataset, val_dataset)
    """
    image_dir = image_dir or os.path.join(root_dir, "images")
    data_file = data_file or os.path.join(root_dir, "data/image_labels.pqt")
    dataset_splits = dataset_splits or {"train": [], "val": [], "test": []}

    ensure_data_file(data_file)
    logger.info(f"📊 使用数据文件: {data_file}")
    logger.info(f"🚀 数据加载方式: {'Zarr直接加载' if use_zarr else '图像文件加载'}")
    if use_zarr:
        logger.info(f"🎯 Zarr marker: {zarr_marker}")

    transform_train, transform_eval = get_default_transforms()

    train_dataset = PatchImageDataset(
        parquet_path=data_file,
        root_dir=image_dir,
        patch_size=patch_size,
        transform=transform_train,
        subset=dataset_splits.get("train") or None,
        cache_images=False,
        target_biomarkers=target_biomarkers,
        use_zarr=use_zarr,
        zarr_marker=zarr_marker,
    )

    val_dataset = PatchImageDataset(
        parquet_path=data_file,
        root_dir=image_dir,
        patch_size=patch_size,
        transform=transform_eval,
        subset=dataset_splits.get("val") or None,
        cache_images=True,
        target_biomarkers=target_biomarkers,
        use_zarr=use_zarr,
        zarr_marker=zarr_marker,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 4,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, train_dataset, val_dataset


class ProteinPredictor:
    """High-level trainer/evaluator for protein prediction from H&E patches.

    This class owns the model, data loaders, and training loop. The defaults are
    chosen to mirror the previously standalone training script behavior.

    Attributes are kept explicit to favor clarity and debuggability.
    """

    # Default configuration constants (can be overridden in __init__ args)
    DEFAULT_TARGET_BIOMARKERS: List[str] = ["CD3", "CD8", "CD20", "CD68"]

    def __init__(
        self,
        *,
        root_dir: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 1e-4,
        eval_interval: int = 3000,
        patience: int = 75000,
    ) -> None:
        """Initialize trainer with paths, hyperparameters, and dataset metadata.

        Args:
            root_dir: 项目根目录，用于保存 `runs/` 等输出。
            train_loader: 已准备好的训练 DataLoader。
            val_loader: 已准备好的验证 DataLoader。
            learning_rate: 学习率。
            eval_interval: 多少步进行一次验证。
            patience: 早停步数（自上次最佳起累计步数）。
        """

        self.root_dir = root_dir

        self.learning_rate = learning_rate
        self.eval_interval = eval_interval
        self.patience = patience
        
        # Data related members
        self.train_loader: DataLoader = train_loader
        self.val_loader: DataLoader = val_loader

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🔧 Using device: {self.device}")

        num_outputs = self._infer_num_outputs_from_loader(train_loader)
        self.model = self.build_model(num_outputs=num_outputs).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="max", factor=0.5, patience=10, verbose=True)
        self.criterion = self.masked_mse_loss

    # =====================
    # Model & Loss utilities
    # =====================
    @staticmethod
    def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute masked mean squared error.

        Args:
            pred: Prediction tensor of shape (N, D).
            target: Target tensor of shape (N, D).
            mask: Boolean/binary tensor of shape (N, D), 1 indicates valid.

        Returns:
            Scalar loss tensor.
        """
        mask_bool = mask.bool()
        masked_pred = torch.masked_select(pred, mask_bool)
        masked_target = torch.masked_select(target, mask_bool)
        return F.mse_loss(masked_pred, masked_target, reduction="mean")

    @staticmethod
    def build_model(num_outputs: int) -> nn.Module:
        """Create the base ConvNeXt-Small model with custom output head.

        Args:
            num_outputs: Number of regression outputs.

        Returns:
            Initialized `nn.Module` ready for training.
        """
        model = models.convnext_small(weights="IMAGENET1K_V1")
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_outputs)
        return model

    # =====================
    # Data helpers
    # =====================
    def _infer_num_outputs_from_loader(self, loader: DataLoader) -> int:
        """从 DataLoader 的一个样本推断输出维度。"""
        dataset = loader.dataset
        assert len(dataset) > 0, "Train dataset unable to load"
        sample = dataset[0]
        assert sample is not None, "Invalid sample from train dataset"
        _, exp_vec, _, _, _, _ = sample
        num_outputs = len(exp_vec)
        logger.info(f"🎯 根据数据集实际格式确定输出维度: {num_outputs}")
        return int(num_outputs)

    @torch.no_grad()
    def evaluate(
        self,
        *,
        data_loader: DataLoader,
        save_predictions: bool = False,
        save_filename: str = "predictions.pqt",
        pred_biomarkers: Optional[List[str]] = None,
    ) -> float:
        """在验证集上计算全局 masked MSE，可选保存预测。

        Args:
            data_loader: 要评估的数据集 loader。
            save_predictions: 是否保存预测到 parquet。
            save_filename: 保存文件名。
            pred_biomarkers: 预测输出名称列表，默认与数据集一致。

        Returns:
            全局 masked MSE（float）。
        """
        assert self.model is not None
        model = self.model
        model.eval()

        # 推断标签名
        if pred_biomarkers is None:
            assert getattr(data_loader.dataset, "target_biomarkers", None) is not None
            bm_labels = data_loader.dataset.target_biomarkers
            pred_biomarkers = bm_labels
        else:
            bm_labels = pred_biomarkers

        total_sq_error = 0.0
        total_count = 0
        rows: List[List[float]] = []

        run_dir = os.path.join(self.root_dir, "runs")
        os.makedirs(run_dir, exist_ok=True)

        for batch in tqdm(data_loader):
            if batch is None:
                continue
                
            inputs, exp_vec, valid_mask, X, Y, img_indices = batch
            inputs = inputs.to(self.device)
            exp_vec = exp_vec.to(self.device)
            valid_mask = valid_mask.to(self.device).bool()

            outputs = model(inputs)
            diff = outputs - exp_vec
            sq = (diff * diff)[valid_mask]
            total_sq_error += float(sq.sum().item())
            total_count += int(valid_mask.sum().item())

            if save_predictions:
                outputs_np = outputs.detach().cpu().numpy()
                exp_vec_np = exp_vec.detach().cpu().numpy()
                for i in range(len(inputs)):
                    rows.append(
                        [X[i].item(), Y[i].item(), img_indices[i].item()] + list(outputs_np[i]) + list(exp_vec_np[i])
                    )

        val_mse = total_sq_error / max(total_count, 1)

        if save_predictions and rows:
            eval_df = pd.DataFrame(
                rows,
                columns=["X", "Y", "image_idx"]
                + [f"pred_{x}" for x in pred_biomarkers]
                + [f"gt_{x}" for x in bm_labels],
            )
            out_path = os.path.join(run_dir, save_filename)
            eval_df.to_parquet(out_path)

        return float(val_mse)

    def train(self) -> None:
        """Execute the training loop with periodic evaluation and early stopping."""
        # Training state
        step = 0
        best_val_mse = float("inf")
        steps_since_best = 0

        assert self.train_loader is not None
        while True:
            self.model.train()
            for batch in self.train_loader:
                if batch is None:
                    continue
                    
                inputs, labels, mask, _, _, _ = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                mask = mask.to(self.device)

                assert self.optimizer is not None
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels, mask)

                if torch.isnan(loss):
                    logger.warning("Warning: Loss is NaN, skipping batch")
                    continue

                loss.backward()
                self.optimizer.step()

                if step % 100 == 0:
                    logger.info({"train_loss": float(loss.item())})
                    logger.info(f"📊 Step {step}, Train Loss: {loss.item():.6f}")

                # Eval
                if step % self.eval_interval == 0:
                    logger.info(f"🔍 Start Validation (Step {step})...")
                    val_mse = self.evaluate(data_loader=self.val_loader, save_predictions=False, save_filename=f"predictions_{step}.pqt")

                    logger.info(f"📈 Validation mse: {val_mse:.6f}")

                    # Save best
                    run_dir = os.path.join(self.root_dir, "runs")
                    os.makedirs(run_dir, exist_ok=True)
                    if val_mse < best_val_mse:
                        best_val_mse = val_mse
                        torch.save(self.model.state_dict(), os.path.join(run_dir, "best_model.pth"))
                        steps_since_best = 0
                    else:
                        steps_since_best += self.eval_interval

                    assert self.scheduler is not None
                    self.scheduler.step(val_mse)

                    # Early stopping
                    if steps_since_best >= self.patience:
                        logger.info(f"Early stopping after {step} steps")
                        return
                step += 1


__all__ = ["ProteinPredictor"]


