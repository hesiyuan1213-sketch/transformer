# -*- coding: utf-8 -*-
"""
超参对比实验脚本。
在多个超参配置下进行训练，并对比结果。
用法：python hyperparameter_experiment.py
"""
import argparse
import csv
import json
import os
import sys
from typing import Dict, List

import torch

from train import (
    get_device,
    set_seed,
    eval_loss,
    TRAIN_PT,
    VAL_PT,
    VOCAB,
    WEIGHT_DECAY,
    GRAD_CLIP_NORM,
    DATALOADER_NUM_WORKERS,
    DATALOADER_DROP_LAST,
)
from dataset import PoetryBlockDataset, load_vocab_json
from model import CharGPT
from torch.utils.data import DataLoader


def run_single_config(
    config_name: str,
    config: Dict,
    device: torch.device,
    vocab_size: int,
    train_ds,
    val_ds,
) -> Dict:
    """
    运行单个超参配置的训练。
    返回最终的验证 loss 和其他指标。
    """
    print(f"\n{'=' * 80}")
    print(f"运行配置: {config_name}")
    print(f"{'=' * 80}")
    print(json.dumps(config, indent=2, ensure_ascii=False))
    print()

    set_seed(config.get("seed", 42))

    # 创建数据加载器
    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=DATALOADER_NUM_WORKERS,
        drop_last=DATALOADER_DROP_LAST,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=DATALOADER_NUM_WORKERS,
        drop_last=DATALOADER_DROP_LAST,
    )

    # 创建模型
    model = CharGPT(
        vocab_size=vocab_size,
        block_size=config["block_size"],
        d_model=config["d_model"],
        n_head=config["n_head"],
        n_layer=config["n_layer"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {param_count:,}")

    # 创建优化器
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=WEIGHT_DECAY)

    # 训练循环
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for ep in range(1, config["epochs"] + 1):
        model.train()
        run_loss = 0.0
        cnt = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()
            run_loss += loss.item() * x.size(0)
            cnt += x.size(0)

            if (batch_idx + 1) % max(1, len(train_loader) // 3) == 0:
                print(f"  Epoch {ep}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_train_loss = run_loss / max(cnt, 1)
        avg_val_loss = eval_loss(model, val_loader, device, max_batches=200)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch {ep}/{config['epochs']} | train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"  ✓ 更优验证 loss: {best_val_loss:.4f}")

    return {
        "config_name": config_name,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
        "best_val_loss": best_val_loss,
        "param_count": param_count,
        "config": config,
    }


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    # 检查必要文件
    for f in (TRAIN_PT, VAL_PT, VOCAB):
        if not os.path.isfile(f):
            print(f"缺少 {f}，请先运行 prepare_data.py", file=sys.stderr)
            sys.exit(1)

    device = get_device()
    print("设备:", device)

    # 加载数据和词表
    _, __, vocab_size = load_vocab_json(VOCAB)
    print("正在加载训练/验证张量...", flush=True)

    # 定义要对比的超参配置（至少 3 组，涉及至少 2 类不同超参）
    # 以下是示例配置，您可以根据需要修改
    configurations = {
        "配置1-基础": {
            "block_size": 128,
            "d_model": 256,
            "n_head": 8,
            "n_layer": 4,
            "d_ff": 1024,
            "dropout": 0.1,
            "batch_size": 32,
            "lr": 3e-4,
            "epochs": 2,
            "seed": 42,
            "train_num_samples": 20000,
        },
        "配置2-较大模型": {
            "block_size": 128,
            "d_model": 384,  # 增加 d_model
            "n_head": 8,
            "n_layer": 6,     # 增加层数
            "d_ff": 1536,
            "dropout": 0.1,
            "batch_size": 32,
            "lr": 3e-4,
            "epochs": 2,
            "seed": 42,
            "train_num_samples": 20000,
        },
        "配置3-高学习率": {
            "block_size": 128,
            "d_model": 256,
            "n_head": 8,
            "n_layer": 4,
            "d_ff": 1024,
            "dropout": 0.1,
            "batch_size": 32,
            "lr": 5e-4,       # 提高学习率
            "epochs": 2,
            "seed": 42,
            "train_num_samples": 20000,
        },
        "配置4-高dropout": {
            "block_size": 128,
            "d_model": 256,
            "n_head": 8,
            "n_layer": 4,
            "d_ff": 1024,
            "dropout": 0.2,   # 增加dropout
            "batch_size": 32,
            "lr": 3e-4,
            "epochs": 2,
            "seed": 42,
            "train_num_samples": 20000,
        },
    }

    print(f"\n准备进行 {len(configurations)} 组配置的对比实验\n")

    # 加载数据集（这里只加载一次）
    train_ds = PoetryBlockDataset(
        TRAIN_PT,
        block_size=128,  # 使用最大的 block_size
        num_samples=20000,
        sample_random=True,
    )
    val_ds = PoetryBlockDataset(
        VAL_PT,
        block_size=128,
        num_samples=None,
        sample_random=False,
    )

    results = []

    for config_name, config in configurations.items():
        try:
            result = run_single_config(
                config_name,
                config,
                device,
                vocab_size,
                train_ds,
                val_ds,
            )
            results.append(result)
        except Exception as e:
            print(f"配置 {config_name} 出错: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    # 保存结果
    output_csv = "hyperparameter_results.csv"
    output_txt = "hyperparameter_results.txt"

    with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["配置名", "最终训练Loss", "最终验证Loss", "最优验证Loss", "参数量", "d_model", "n_layer", "lr", "dropout"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "配置名": r["config_name"],
                "最终训练Loss": f"{r['final_train_loss']:.4f}" if r['final_train_loss'] else "N/A",
                "最终验证Loss": f"{r['final_val_loss']:.4f}" if r['final_val_loss'] else "N/A",
                "最优验证Loss": f"{r['best_val_loss']:.4f}",
                "参数量": r["param_count"],
                "d_model": r["config"]["d_model"],
                "n_layer": r["config"]["n_layer"],
                "lr": r["config"]["lr"],
                "dropout": r["config"]["dropout"],
            })

    with open(output_txt, "w", encoding="utf-8") as txt_file:
        txt_file.write("=" * 100 + "\n")
        txt_file.write("超参对比实验报告\n")
        txt_file.write("=" * 100 + "\n\n")

        # 汇总表
        txt_file.write("汇总表:\n")
        txt_file.write("-" * 100 + "\n")
        txt_file.write(f"{'配置名':<20} {'最终训练Loss':<15} {'最终验证Loss':<15} {'最优验证Loss':<15} {'参数量':<12}\n")
        txt_file.write("-" * 100 + "\n")

        best_config = min(results, key=lambda x: x["best_val_loss"])

        for r in results:
            train_l = f"{r['final_train_loss']:.4f}" if r['final_train_loss'] else "N/A"
            val_l = f"{r['final_val_loss']:.4f}" if r['final_val_loss'] else "N/A"
            best_l = f"{r['best_val_loss']:.4f}"
            mark = " ← 最优" if r == best_config else ""
            txt_file.write(f"{r['config_name']:<20} {train_l:<15} {val_l:<15} {best_l:<15} {r['param_count']:<12}{mark}\n")

        txt_file.write("-" * 100 + "\n\n")

        # 详细配置信息
        txt_file.write("\n详细配置:\n")
        txt_file.write("=" * 100 + "\n")
        for r in results:
            txt_file.write(f"\n{r['config_name']}:\n")
            for key, value in r["config"].items():
                txt_file.write(f"  {key}: {value}\n")
            txt_file.write(f"  参数量: {r['param_count']:,}\n")
            txt_file.write(f"  最优验证 Loss: {r['best_val_loss']:.4f}\n")

        # 分析
        txt_file.write("\n" + "=" * 100 + "\n")
        txt_file.write("分析:\n")
        txt_file.write("=" * 100 + "\n")
        txt_file.write(f"最优配置: {best_config['config_name']}\n")
        txt_file.write(f"最优验证 Loss: {best_config['best_val_loss']:.4f}\n")
        txt_file.write(f"参数量: {best_config['param_count']:,}\n\n")

        txt_file.write("观察:\n")
        for r in results:
            if r != best_config:
                diff = ((r["best_val_loss"] - best_config["best_val_loss"]) / best_config["best_val_loss"]) * 100
                txt_file.write(f"- {r['config_name']}: 验证Loss比最优配置高 {diff:.1f}%\n")

    print(f"\n✓ 结果已保存到:")
    print(f"  - CSV 格式: {os.path.abspath(output_csv)}")
    print(f"  - 文本格式: {os.path.abspath(output_txt)}")
    print(f"\n最优配置: {best_config['config_name']}")
    print(f"最优验证 Loss: {best_config['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()
