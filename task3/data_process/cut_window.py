#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 candidates_win{W}.csv 批量裁剪 W 秒窗口，按 5 段摆放策略导出 wav，并生成 manifest.csv。

与原版差异：将 **seg2** 和 **seg4** 的摆放策略改为“居中”（center）。

输入 CSV（来自 collect_window_candidates_both.py）应包含：
source,file,abs_audio_path,onset,offset,class,left_extend_to,right_extend_to,
event_duration,left_space,right_space,window_sec,segment_sec

摆放规则（本版）：
  seg1: 事件在第1段（[0,seg)）的最右侧（事件 offset 对齐到 seg） -> right_edge
  seg2: 事件在第2段（[seg,2seg)）的中间     （事件中心对齐到 1.5*seg） -> center
  seg3: 事件在第3段（[2seg,3seg)）的中间     （事件中心对齐到 2.5*seg） -> center
  seg4: 事件在第4段（[3seg,4seg)）的中间     （事件中心对齐到 3.5*seg） -> center
  seg5: 事件在第5段（[4seg,5seg=W)）的最左侧（事件 onset 对齐到 4*seg） -> left_edge

输出目录层级：
out_dir/
  win10s/
    seg1/class_*/...
    ...
  win20s/
  ...

输出 manifest.csv 列：
new_out_path,class,event_onset_sec,event_offset_sec,
orig_file,orig_onset_sec,orig_offset_sec,
window_sec,segment_index,placement,window_start_sec,window_end_sec,
sample_rate,channels,source

依赖：
  pip install pandas pydub
  # 需要系统安装 ffmpeg
"""

import os
import math
import argparse
import pandas as pd
from typing import Dict, List, Tuple
from pydub import AudioSegment

EPS = 1e-6

# ---- 计算每段的“理想”窗口起点（start），使得事件在 W 秒窗口内满足指定摆放 ----

def ideal_window_start_for_segment(onset: float, offset: float, W: float, seg: float, seg_idx: int) -> Tuple[float, str]:
    """
    返回 (ideal_start_sec, placement_tag)
    seg_idx: 1..5
    改动点：seg2 与 seg4 的 target 位置改为该段的中心（1.5*seg、3.5*seg）。
    """
    d = offset - onset
    if seg_idx == 1:
        # 事件 offset 对齐到 seg（第1段最右侧）
        event_offset_in_win = seg
        event_onset_in_win = event_offset_in_win - d
        start = onset - event_onset_in_win
        return start, "right_edge"
    elif seg_idx == 2:
        # 事件中心对齐到 1.5*seg（第2段居中）
        center = 1.5 * seg
        event_onset_in_win = center - d / 2.0
        start = onset - event_onset_in_win
        return start, "center"
    elif seg_idx == 3:
        # 事件中心对齐到 2.5*seg（第3段居中）
        center = 2.5 * seg
        event_onset_in_win = center - d / 2.0
        start = onset - event_onset_in_win
        return start, "center"
    elif seg_idx == 4:
        # 事件中心对齐到 3.5*seg（第4段居中）
        center = 3.5 * seg
        event_onset_in_win = center - d / 2.0
        start = onset - event_onset_in_win
        return start, "center"
    elif seg_idx == 5:
        # 事件 onset 对齐到 4*seg（第5段最左侧）
        event_onset_in_win = 4.0 * seg
        start = onset - event_onset_in_win
        return start, "left_edge"
    else:
        raise ValueError("seg_idx must be in [1..5]")

def clamp_window_start(ideal_start: float, onset: float, offset: float,
                       left_extend_to: float, right_extend_to: float, W: float) -> float:
    """
    将窗口起点夹到可行范围：
      - start >= left_extend_to
      - start + W <= right_extend_to
      - 窗口必须包含事件：start <= onset 且 start >= offset - W
    """
    lo = max(left_extend_to, offset - W)
    hi = min(onset, right_extend_to - W)
    # 在前置筛选条件下，理想起点应当位于 [lo, hi] 区间；若不在，进行夹紧（极端容错）
    if lo > hi:
        # 退而求其次，贴边
        return max(left_extend_to, min(onset, right_extend_to - W))
    return min(max(ideal_start, lo), hi)

# ---- I/O & 剪切 ----

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def cut_and_save(audio_path: str, start_sec: float, W: float, save_path: str) -> Tuple[float, int, int]:
    """
    剪切 [start_sec, start_sec + W] 并保存到 save_path。
    返回 (duration_sec, sample_rate, channels) —— 这里 duration_sec 应为 W。
    """
    seg = AudioSegment.from_file(audio_path)
    sr = seg.frame_rate
    ch = seg.channels

    start_ms = int(round(start_sec * 1000))
    end_ms = int(round((start_sec + W) * 1000))

    # 安全边界（理论上已保证）
    n = len(seg)
    if start_ms < 0 or end_ms > n:
        raise ValueError(f"Window out of bounds: [{start_sec}, {start_sec + W}] vs file {n/1000.0}s")

    clip = seg[start_ms:end_ms]
    ensure_dir(os.path.dirname(save_path))
    clip.export(save_path, format="wav")
    return (len(clip) / 1000.0), sr, ch

# ---- 主流程 ----

def process_one_csv(csv_path: str, out_dir: str, override_window: float = None) -> List[Dict]:
    """
    处理单个 candidates_winW.csv，返回 manifest 的行列表。
    如果 override_window 不为空，会使用它（通常无需设置）。
    """
    df = pd.read_csv(csv_path)
    req = {"source","file","abs_audio_path","onset","offset","class","left_extend_to","right_extend_to","window_sec","segment_sec"}
    if not req.issubset(df.columns):
        raise ValueError(f"{csv_path} 缺少列：{req - set(df.columns)}")

    rows = []
    for idx, r in df.iterrows():
        src     = str(r["source"])
        frel    = str(r["file"])
        apath   = str(r["abs_audio_path"])
        onset   = float(r["onset"])
        offset  = float(r["offset"])
        cls     = int(r["class"])
        left_to = float(r["left_extend_to"])
        right_to= float(r["right_extend_to"])
        W       = float(r["window_sec"] if override_window is None else override_window)
        seg     = float(r["segment_sec"]) if pd.notnull(r["segment_sec"]) else W / 5.0

        for seg_idx in range(1, 6):
            ideal_start, placement = ideal_window_start_for_segment(onset, offset, W, seg, seg_idx)
            start_sec = clamp_window_start(ideal_start, onset, offset, left_to, right_to, W)
            end_sec   = start_sec + W

            # 事件在新窗口内的真实位置（四舍五入到 0.1s，便于评测）
            event_on_in_new  = round(onset  - start_sec + 1e-9, 1)
            event_off_in_new = round(offset - start_sec + 1e-9, 1)

            # 输出路径：out_dir/win{W}s/seg{seg_idx}/class_{cls}/{source}_{base}_{start}-{end}.wav
            base  = os.path.splitext(os.path.basename(frel))[0]
            save_dir = os.path.join(out_dir, f"win{int(W)}s", f"seg{seg_idx}", f"class_{cls}")
            fname = f"{src}_{base}_win{int(W)}s_{start_sec:.3f}-{end_sec:.3f}.wav".replace("/", "_")
            save_path = os.path.join(save_dir, fname)

            try:
                dur_sec, sr, ch = cut_and_save(apath, start_sec, W, save_path)
            except Exception as e:
                print(f"[SKIP] {apath} at [{start_sec:.3f},{end_sec:.3f}] -> {e}")
                continue

            rows.append({
                "new_out_path": save_path,
                "class": cls,
                "event_onset_sec": event_on_in_new,
                "event_offset_sec": event_off_in_new,
                "orig_file": frel,
                "orig_onset_sec": round(onset, 1),
                "orig_offset_sec": round(offset, 1),
                "window_sec": int(W),
                "segment_index": seg_idx,
                "placement": placement,  # right_edge / center / left_edge
                "window_start_sec": round(start_sec, 3),
                "window_end_sec": round(end_sec, 3),
                "sample_rate": sr,
                "channels": ch,
                "source": src
            })

            print(f"[OK] {save_path} | seg{seg_idx}@{placement} | "
                  f"event {event_on_in_new:.1f}-{event_off_in_new:.1f}s")

    return rows

def main():
    parser = argparse.ArgumentParser(description="按 5 段摆放从候选 CSV 批量裁剪窗口音频并生成 manifest（seg2/seg4 居中版）")
    parser.add_argument("--candidates-dir", required=True,
                        help="包含 candidates_win10.csv ... candidates_win60.csv 的目录")
    parser.add_argument("--out-dir", default="win_cuts", help="输出根目录（分窗口/分段/分类存储）")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    manifest_rows: List[Dict] = []
    for W in [10, 20, 30, 40, 50, 60]:
        csv_path = os.path.join(args.candidates_dir, f"candidates_win{W}.csv")
        if not os.path.exists(csv_path):
            print(f"[WARN] 缺少 {csv_path}，跳过该窗口。")
            continue
        print(f"[RUN] {csv_path}")
        manifest_rows.extend(process_one_csv(csv_path, args.out_dir, override_window=float(W)))

    if manifest_rows:
        manifest_path = os.path.join(args.out_dir, "manifest.csv")
        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
        print(f"[DONE] 写出 manifest：{manifest_path} （共 {len(manifest_rows)} 条）")
    else:
        print("[INFO] 没有生成任何音频片段。")

if __name__ == "__main__":
    main()
