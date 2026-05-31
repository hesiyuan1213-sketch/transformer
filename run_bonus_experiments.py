# -*- coding: utf-8 -*-
"""
加分项自动实验脚本。
前提：已运行 prepare_data.py，当前目录存在 vocab.json、train_data.pt、val_data.pt。
功能：
1. 调参对比：训练 3 组超参，记录最优验证集 loss。
2. Temperature 对比：固定起始字与 max_new，每个 temperature 生成 3 次。
输出：bonus_results/hparam_results.csv、bonus_results/temperature_samples.csv。
"""
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "bonus_results"
OUT.mkdir(exist_ok=True)

# 可按机器性能调整：为了课堂加分项，建议保证三组训练量相同
TRAIN_COMMON = [
    "--epochs", "2",
    "--train_num_samples", "20000",
    "--max_train_batches", "300",
    "--val_batches", "50",
    "--batch_size", "16",
    "--block_size", "128",
    "--seed", "42",
]

HPARAM_CONFIGS = [
    {"name": "A_baseline", "d_model": 128, "n_head": 4, "n_layer": 2, "d_ff": 512, "lr": "3e-4"},
    {"name": "B_wider", "d_model": 192, "n_head": 6, "n_layer": 2, "d_ff": 768, "lr": "3e-4"},
    {"name": "C_higher_lr", "d_model": 128, "n_head": 4, "n_layer": 2, "d_ff": 512, "lr": "1e-3"},
]

TEMPERATURES = [0.4, 0.8, 1.2]
PROMPT = "春"
MAX_NEW = 80
RUNS_PER_TEMP = 3


def run(cmd, log_path):
    with open(log_path, "w", encoding="utf-8") as f:
        p = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"命令失败，详见 {log_path}: {' '.join(cmd)}")
    return Path(log_path).read_text(encoding="utf-8", errors="ignore")


def parse_best_val(text):
    vals = [float(x) for x in re.findall(r"val_loss=([0-9]+(?:\.[0-9]+)?)", text)]
    if vals:
        return min(vals)
    m = re.search(r"最佳 val_loss.*?:\s*([0-9]+(?:\.[0-9]+)?)", text)
    return float(m.group(1)) if m else None


def main():
    required = [ROOT / "vocab.json", ROOT / "train_data.pt", ROOT / "val_data.pt"]
    missing = [str(p.name) for p in required if not p.exists()]
    if missing:
        print("缺少数据文件：" + ", ".join(missing))
        print("请先运行：python prepare_data.py")
        sys.exit(1)

    hparam_rows = []
    best_ckpt = None
    best_val = float("inf")
    for cfg in HPARAM_CONFIGS:
        ckpt = OUT / f"ckpt_{cfg['name']}.pt"
        plot = OUT / f"loss_{cfg['name']}.png"
        log = OUT / f"train_{cfg['name']}.log"
        cmd = [
            sys.executable, "train.py",
            *TRAIN_COMMON,
            "--d_model", str(cfg["d_model"]),
            "--n_head", str(cfg["n_head"]),
            "--n_layer", str(cfg["n_layer"]),
            "--d_ff", str(cfg["d_ff"]),
            "--lr", str(cfg["lr"]),
            "--save", str(ckpt),
            "--log_plot", str(plot),
        ]
        print("训练配置：", cfg["name"])
        text = run(cmd, log)
        val = parse_best_val(text)
        hparam_rows.append({**cfg, "epochs": 2, "max_train_batches": 300, "val_loss": val, "ckpt": str(ckpt.name), "loss_curve": str(plot.name)})
        if val is not None and val < best_val:
            best_val = val
            best_ckpt = ckpt

    with open(OUT / "hparam_results.csv", "w", encoding="utf-8", newline="") as f:
        fieldnames = ["name", "d_model", "n_head", "n_layer", "d_ff", "lr", "epochs", "max_train_batches", "val_loss", "ckpt", "loss_curve"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(hparam_rows)

    if best_ckpt is None:
        raise RuntimeError("没有得到可用 checkpoint，无法进行 temperature 实验")

    sample_rows = []
    for temp in TEMPERATURES:
        for run_id in range(1, RUNS_PER_TEMP + 1):
            log = OUT / f"sample_temp_{temp}_run_{run_id}.log"
            cmd = [
                sys.executable, "predict.py",
                "--once",
                "--ckpt", str(best_ckpt),
                "--prompt", PROMPT,
                "--max_new", str(MAX_NEW),
                "--temperature", str(temp),
            ]
            text = run(cmd, log)
            sample = text.split("续写:", 1)[-1].strip() if "续写:" in text else text.strip()
            sample_rows.append({"temperature": temp, "run": run_id, "prompt": PROMPT, "max_new": MAX_NEW, "sample": sample})

    with open(OUT / "temperature_samples.csv", "w", encoding="utf-8", newline="") as f:
        fieldnames = ["temperature", "run", "prompt", "max_new", "sample"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(sample_rows)

    print("加分项实验完成，结果保存在：", OUT)


if __name__ == "__main__":
    main()
