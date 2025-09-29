#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
筛选规则（去掉‘纯净’要求）：
- A组：{4,12,7,6,2,11,3}，要求 duration < 3
- B组：{0,1,8}：
    - 子类1：duration < 3
    - 子类2：7 < duration < 10
- 找到所有符合的数据（不做每类上限与采样），与是否与其他类别重叠无关
- 输出按 class 升序，类内按 file、onset 排序
"""

import argparse
import pandas as pd

# 目标类别集合
GROUP_LT3 = {4, 12, 7, 6, 2, 11, 3}
GROUP_0_1_8 = {0, 1, 8}
TARGET_CLASSES = sorted(list(GROUP_LT3 | GROUP_0_1_8))

def apply_rules_no_isolation(df: pd.DataFrame) -> pd.DataFrame:
    """
    按新规则（无‘纯净/无重叠’要求）筛选：
      - A组：{4,12,7,6,2,11,3} 且 duration < 3
      - B组1：{0,1,8} 且 duration < 3
      - B组2：{0,1,8} 且 8 < duration < 12
    """
    df = df.copy()
    df["duration"] = df["offset"] - df["onset"]

    # 只保留目标类别
    df = df[df["class"].isin(TARGET_CLASSES)]

    # A组：小于3秒
    mask_A = df["class"].isin(GROUP_LT3) & (df["duration"] < 3.0)

    # B组两类
    mask_B1 = df["class"].isin(GROUP_0_1_8) & (df["duration"] < 3.0)
    mask_B2 = df["class"].isin(GROUP_0_1_8) & (df["duration"] > 7.0) & (df["duration"] < 10.0)

    # 合并
    df_out = df[mask_A | mask_B1 | mask_B2].copy()

    # 输出前去掉临时列
    return df_out.drop(columns=["duration"], errors="ignore")

def process_csv(input_csv: str, output_csv: str) -> None:
    df = pd.read_csv(input_csv)
    required_cols = {"file", "onset", "offset", "class"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"输入CSV缺少列：{missing}")

    # 类型清洗
    df = df.copy()
    df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
    df["offset"] = pd.to_numeric(df["offset"], errors="coerce")
    df["class"] = pd.to_numeric(df["class"], errors="coerce").astype("Int64")

    # 去除缺失与非法时序
    df = df.dropna(subset=["onset", "offset", "class"])
    df = df[df["offset"] > df["onset"]]

    # 应用规则（无‘纯净’限制）
    df = apply_rules_no_isolation(df)

    # 排序与保存
    df = df.sort_values(["class", "file", "onset"]).reset_index(drop=True)
    df.to_csv(output_csv, index=False)
    print(f"已保存筛选结果到：{output_csv}（共 {len(df)} 条）")

def main():
    parser = argparse.ArgumentParser(description="依据规则筛选：仅按类别+时长（不要求纯净）")
    parser.add_argument("input_csv", help="输入 CSV 路径")
    parser.add_argument("output_csv", help="输出 CSV 路径")
    args = parser.parse_args()

    process_csv(args.input_csv, args.output_csv)

if __name__ == "__main__":
    main()
