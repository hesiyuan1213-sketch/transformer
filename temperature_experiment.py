# -*- coding: utf-8 -*-
"""
Temperature 对比实验脚本。
在多个 temperature 值下进行生成，并保存结果为表格和可视化。
用法：python temperature_experiment.py [--ckpt ckpt_best.pt] [--vocab vocab.json]
"""
import argparse
import csv
import os
import sys
from typing import Dict, List

import torch
import torch.nn.functional as F

from dataset import load_vocab_json
from model import CharGPT
from predict import load_for_generate, first_in_vocab_char, generate_one
from train import get_device


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    p = argparse.ArgumentParser(description="Temperature 对比实验")
    p.add_argument("--ckpt", type=str, default="ckpt_best.pt", help="模型检查点路径")
    p.add_argument("--vocab", type=str, default="vocab.json", help="词表路径")
    p.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[0.4, 0.7, 1.0, 1.3],
        help="要测试的 temperature 值列表，默认 [0.4, 0.7, 1.0, 1.3]"
    )
    p.add_argument("--prompt", type=str, default="春", help="起笔字")
    p.add_argument("--max_new", type=int, default=100, help="新增长度（字符数）")
    p.add_argument("--num_samples", type=int, default=3, help="每个 temperature 下生成的样本数")
    p.add_argument(
        "--output_csv",
        type=str,
        default="temperature_results.csv",
        help="结果输出CSV文件"
    )
    p.add_argument(
        "--output_txt",
        type=str,
        default="temperature_results.txt",
        help="结果输出文本文件（用于报告）"
    )
    args = p.parse_args()

    if not os.path.isfile(args.ckpt):
        print("未找到:", args.ckpt, "请先训练: python train.py", file=sys.stderr)
        sys.exit(1)

    print("正在加载模型与词表…", flush=True)
    stoi, itos, _ = load_vocab_json(args.vocab)
    device = get_device()
    model, _hp = load_for_generate(args.ckpt, device)
    print("设备:", device, flush=True)
    print(f"参数: 起笔='{args.prompt}', max_new={args.max_new}, num_samples={args.num_samples}")
    print(f"Temperature 列表: {args.temperatures}\n")

    # 验证起笔字在词表中
    try:
        start_char = first_in_vocab_char(args.prompt, stoi)
    except ValueError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)

    results = []
    
    with open(args.output_txt, "w", encoding="utf-8") as txt_file:
        txt_file.write("=" * 80 + "\n")
        txt_file.write("Temperature 对比实验报告\n")
        txt_file.write("=" * 80 + "\n\n")
        txt_file.write(f"模型检查点: {args.ckpt}\n")
        txt_file.write(f"起笔字: '{start_char}'\n")
        txt_file.write(f"新增长度: {args.max_new}\n")
        txt_file.write(f"每个温度的样本数: {args.num_samples}\n\n")

        for temp in args.temperatures:
            txt_file.write(f"\n{'=' * 80}\n")
            txt_file.write(f"Temperature = {temp}\n")
            txt_file.write(f"{'=' * 80}\n\n")
            
            for sample_idx in range(1, args.num_samples + 1):
                try:
                    text = generate_one(
                        model,
                        stoi,
                        itos,
                        device,
                        start_char,
                        args.max_new,
                        temp,
                        stop_newline=False,
                    )
                except Exception as e:
                    text = f"[生成失败: {e}]"

                results.append({
                    "Temperature": temp,
                    "Sample": sample_idx,
                    "生成文本": text,
                })

                txt_file.write(f"样本 {sample_idx}:\n{text}\n\n")
                print(f"[Temp={temp}, Sample={sample_idx}] ✓")

    # 保存 CSV 格式结果
    with open(args.output_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["Temperature", "Sample", "生成文本"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✓ 结果已保存到:")
    print(f"  - 文本格式: {os.path.abspath(args.output_txt)}")
    print(f"  - CSV 格式: {os.path.abspath(args.output_csv)}")
    print(f"\n生成共 {len(results)} 条文本。")
    
    # 简单的观察统计
    print("\n" + "=" * 80)
    print("观察统计:")
    print("=" * 80)
    for temp in args.temperatures:
        samples = [r for r in results if r["Temperature"] == temp]
        if samples:
            avg_len = sum(len(s["生成文本"]) for s in samples) / len(samples)
            print(f"Temperature {temp}: 平均长度 {avg_len:.1f} 字符")
    print("\n提示：较低的 temperature 通常产生更连贯但重复性更高的文本；")
    print("      较高的 temperature 通常产生更多样但可能不连贯的文本。")


if __name__ == "__main__":
    main()
