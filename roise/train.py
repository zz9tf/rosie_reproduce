#!/usr/bin/env python3
"""
优化的训练脚本 - 直接使用分割后的数据集文件
"""

import argparse
import os
from patch_dataset import PatchImageDataset, get_default_transforms, collate_fn
from torch.utils.data import DataLoader
from model import ProteinPredictor

def main():
    """优化的训练主函数"""
    parser = argparse.ArgumentParser(description="优化的基于Patch的H&E图像训练脚本")
    parser.add_argument("--root", required=True, help="项目根目录")
    parser.add_argument("--data-file", required=True, help="数据parquet文件路径")
    parser.add_argument("--target-biomarkers", nargs="+", default=['HE', 'CD3', 'CD8'], help="目标生物标记物")
    parser.add_argument("--batch-size", type=int, default=32, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--eval-interval", type=int, default=1000, help="验证间隔")
    parser.add_argument("--patience", type=int, default=5000, help="早停耐心值")
    parser.add_argument("--num-workers", type=int, default=4, help="数据加载工作进程数")
    parser.add_argument("--patch-size", type=int, default=128, help="patch大小")
    parser.add_argument("--use-zarr", action="store_true", default=True, help="使用zarr直接加载")
    parser.add_argument("--zarr-marker", type=str, default="HE", help="zarr marker名称")
    parser.add_argument("--splits-dir", type=str, default="./splits", help="分割数据集目录")
    
    args = parser.parse_args()
    
    print("🚀 开始优化训练")
    print("=" * 60)
    print(f"🚀 数据加载方式: {'Zarr直接加载' if args.use_zarr else '图像文件加载'}")
    print(f"🎯 Zarr marker: {args.zarr_marker}")
    print(f"📁 分割数据集目录: {args.splits_dir}")
    
    # 检查分割文件是否存在
    train_file = os.path.join(args.splits_dir, "train.parquet")
    val_file = os.path.join(args.splits_dir, "val.parquet")
    test_file = os.path.join(args.splits_dir, "test.parquet")
    
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"训练集文件不存在: {train_file}")
    if not os.path.exists(val_file):
        raise FileNotFoundError(f"验证集文件不存在: {val_file}")
    
    print(f"\n📂 使用分割数据集:")
    print(f"   - 训练集: {train_file}")
    print(f"   - 验证集: {val_file}")
    if os.path.exists(test_file):
        print(f"   - 测试集: {test_file}")
    
    # 创建数据集
    print("\n📦 创建数据集...")
    transform_train, transform_eval = get_default_transforms()
    
    train_dataset = PatchImageDataset(
        parquet_path=train_file,
        patch_size=args.patch_size,
        transform=transform_train,
        cache_images=False,  # 训练时不缓存，节省内存
        target_biomarkers=args.target_biomarkers,
        use_zarr=args.use_zarr,
        zarr_marker=args.zarr_marker,
    )
    
    val_dataset = PatchImageDataset(
        parquet_path=val_file,
        patch_size=args.patch_size,
        transform=transform_eval,
        cache_images=True,  # 验证时缓存，提高速度
        target_biomarkers=args.target_biomarkers,
        use_zarr=args.use_zarr,
        zarr_marker=args.zarr_marker,
    )
    
    print(f"✅ 训练数据集大小: {len(train_dataset)}")
    print(f"✅ 验证数据集大小: {len(val_dataset)}")
    
    # 创建DataLoader
    print("\n🔄 创建DataLoader...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,  # 启用pin_memory加速GPU传输
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,  # 验证时使用更大的批次
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    print(f"✅ 训练批次数: {len(train_loader)}")
    print(f"✅ 验证批次数: {len(val_loader)}")
    
    # 创建训练器
    print("\n🤖 创建训练器...")
    trainer = ProteinPredictor(
        root_dir=args.root,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        eval_interval=args.eval_interval,
        patience=args.patience,
    )
    
    print(f"🔧 使用设备: {trainer.device}")
    print(f"📊 模型输出维度: {trainer.model.classifier[2].out_features}")
    
    # 开始训练
    print("\n🏃 开始训练...")
    print(f"📊 验证间隔: 每 {args.eval_interval} 步")
    print(f"⏰ 早停耐心值: {args.patience} 步")
    
    trainer.train()
    
    print("\n🎉 训练完成！")

if __name__ == "__main__":
    main()


