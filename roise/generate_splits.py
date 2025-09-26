#!/usr/bin/env python3
"""
独立的数据分割索引生成脚本
"""

import argparse
import numpy as np
import pandas as pd

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="生成训练/验证/测试集索引分割")
    parser.add_argument("--data-file", required=True, help="数据parquet文件路径")
    parser.add_argument("--split-ratios", nargs=3, type=float, default=[0.7, 0.15, 0.15], 
                        help="训练/验证/测试集比例 [train val test]")
    parser.add_argument("--split-seed", type=int, default=42, help="数据分割随机种子")
    parser.add_argument("--output", type=str, default="data_splits.npz", help="输出文件名")
    
    args = parser.parse_args()
    
    print("🚀 数据分割索引生成器")
    print("=" * 50)
    print(f"📁 数据文件: {args.data_file}")
    print(f"📊 分割比例: {args.split_ratios}")
    print(f"🎲 随机种子: {args.split_seed}")
    print(f"💾 输出文件: {args.output}")
    print()
    
    # 生成分割
    print("📊 生成数据分割索引...")
    
    # 读取数据文件获取总样本数
    df = pd.read_parquet(args.data_file)
    total_samples = len(df)
    
    print(f"📈 总样本数: {total_samples}")
    print(f"📊 分割比例: 训练={args.split_ratios[0]:.1%}, 验证={args.split_ratios[1]:.1%}, 测试={args.split_ratios[2]:.1%}")
    
    # 验证分割比例
    if abs(sum(args.split_ratios) - 1.0) > 1e-6:
        raise ValueError(f"分割比例之和必须为1.0，当前为: {sum(args.split_ratios)}")
    
    # 生成索引
    indices = np.arange(total_samples)
    
    # 设置随机种子确保可重现性
    np.random.seed(args.split_seed)
    np.random.shuffle(indices)
    
    # 计算分割点
    train_size = int(total_samples * args.split_ratios[0])
    val_size = int(total_samples * args.split_ratios[1])
    
    # 分割索引
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    print(f"✅ 训练集索引: {len(train_indices)} 个样本")
    print(f"✅ 验证集索引: {len(val_indices)} 个样本")
    print(f"✅ 测试集索引: {len(test_indices)} 个样本")
    
    # 保存索引到文件
    np.savez_compressed(
        args.output,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        split_ratios=args.split_ratios,
        split_seed=args.split_seed,
        total_samples=total_samples
    )
    print(f"💾 索引已保存到: {args.output}")
    
    # 保存为CSV格式便于查看
    csv_file = args.output.replace('.npz', '.csv')
    split_df = pd.DataFrame({
        'index': np.concatenate([train_indices, val_indices, test_indices]),
        'split': ['train'] * len(train_indices) + ['val'] * len(val_indices) + ['test'] * len(test_indices)
    })
    split_df = split_df.sort_values('index').reset_index(drop=True)
    split_df.to_csv(csv_file, index=False)
    print(f"📋 CSV格式已保存到: {csv_file}")
    
    print("\n🎉 数据分割完成！")
    print(f"📊 最终统计:")
    print(f"   - 训练集: {len(train_indices)} 个样本 ({len(train_indices)/total_samples:.1%})")
    print(f"   - 验证集: {len(val_indices)} 个样本 ({len(val_indices)/total_samples:.1%})")
    print(f"   - 测试集: {len(test_indices)} 个样本 ({len(test_indices)/total_samples:.1%})")
    print(f"   - 总样本数: {total_samples}")
    print(f"   - 随机种子: {args.split_seed}")
    print(f"   - 分割比例: {args.split_ratios}")

if __name__ == "__main__":
    main()
