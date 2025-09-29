#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Closed-set Multi-label Detection (Form A: set output)
- 切窗 -> 真实存在性集合 -> 询问模型（限定在给定类集合 S 内） -> 模型返回 [id,...] -> 指标
- 模型输出必须是 升序、去重 的整数数组；若无类出现，返回 []

需要的真实标注 CSV 列：
  file,onset,offset,class
  其中 file 与音频 basename 匹配；onset/offset 单位为秒；class 为整型类别ID
"""

import os
import re
import json
import math
import soundfile as sf
import pandas as pd
import numpy as np
import requests
from typing import List, Dict, Set, Tuple, Optional

# ====================== 配置区（按需修改） ======================

# 推理服务
VLLM_ENDPOINT    = "http://127.0.0.1:8011/v1/chat/completions"
MODEL_NAME       = "voxtral-mini-3b"
TEMPERATURE      = 0.2
TOP_P            = 0.95
MAX_TOKENS       = 2048
REQUEST_TIMEOUT  = 300  # seconds

# 窗口长度（秒）
WIN_LENS = [5, 10, 20, 30, 40, 50, 60]

# 类别映射（按你的实验定义）
CLASS_ID_TO_NAME = {
    0: "Female speech, woman speaking",
    1: "Male speech, man speaking",
    2: "Clapping",
    3: "Telephone",
    4: "Laughter",
    5: "Domestic sounds",
    6: "Walk, footsteps",
    7: "Door, open or close",
    8: "Music",
    9: "Musical instrument",
    10: "Water tap, faucet",
    11: "Bell",
    12: "Knock",
}

# 评测的候选类集合 S（仅在这些类内判断是否出现）
TARGET_CLASS_SET = sorted([0,1,2,4])
# 也可以直接用全部 13 类：
# TARGET_CLASS_SET = sorted(CLASS_ID_TO_NAME.keys())

# 数据路径：音频 & 真实标注 CSV（可混合 TAU/SONY；按 basename 匹配）
FILES = [
    "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix001.wav",
    "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix002.wav",
]
EVENTS_CSVS = [
    "/data/user/jzt/crd/audioLLM/train_events/tau.csv",
    "/data/user/jzt/crd/audioLLM/train_events/sony.csv",
]

# 输出模板
DETAIL_CSV_TPL    = "{basename}.win{win:02d}.Aset.detail.csv"
SUMMARY_CSV_TPL   = "{basename}.win{win:02d}.Aset.summary.csv"
REPLIES_JSONL_TPL = "{basename}.win{win:02d}.Aset.replies.jsonl"

# ===============================================================

# ---------------- 音频与切窗 ----------------
def split_fixed_win(wav_path: str, win_len: float):
    wav, sr = sf.read(wav_path)
    if hasattr(wav, "ndim") and wav.ndim > 1:
        wav = wav.mean(axis=1)
    n = len(wav)
    samples = int(round(win_len * sr))
    if samples <= 0:
        raise ValueError(f"Invalid win_len={win_len}")
    wins = []
    i = 0
    while i + samples <= n:
        i0, i1 = i, i + samples
        t0, t1 = i0 / sr, i1 / sr
        wins.append((i0, i1, t0, t1))
        i += samples
    return wav, sr, wins

def array_to_audiochunk(arr, sr):
    import tempfile
    from mistral_common.protocol.instruct.messages import AudioChunk
    from mistral_common.audio import Audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        sf.write(tmp_path, arr, sr, format="wav")
        audio = Audio.from_file(tmp_path, strict=False)
        return AudioChunk.from_audio(audio)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

# ---------------- 真实标签：窗口存在性集合 ----------------
def load_events_csvs(paths: List[str]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            need = {"file","onset","offset","class"}
            missing = need - set(df.columns)
            if missing:
                raise ValueError(f"Events CSV missing columns: {missing} in {p}")
            dfs.append(df[["file","onset","offset","class"]])
    if not dfs:
        raise FileNotFoundError("No valid events CSV found.")
    all_df = pd.concat(dfs, ignore_index=True)
    # 类型清洗
    all_df["file"] = all_df["file"].astype(str)
    all_df["onset"] = all_df["onset"].astype(float)
    all_df["offset"] = all_df["offset"].astype(float)
    all_df["class"] = all_df["class"].astype(int)
    return all_df

def present_classes_in_window(events_df: pd.DataFrame, basename: str, t0: float, t1: float, target_set: Set[int]) -> Set[int]:
    sub = events_df[events_df["file"] == basename]
    if sub.empty:
        return set()
    mask = (sub["offset"] > t0) & (sub["onset"] < t1)
    cats = set(int(c) for c in sub.loc[mask, "class"].tolist())
    return set(c for c in cats if c in target_set)

# ---------------- 提示词 & 模型调用（形态 A：集合输出） ----------------
def render_id_list(target_ids: List[int]) -> str:
    parts = [f"{cid}: {CLASS_ID_TO_NAME.get(cid,'Unknown')}" for cid in target_ids]
    return "[" + ", ".join(parts) + "]"

def build_prompt_setA(target_ids: List[int], win_len: float) -> str:
    return (
        f"This is a {win_len:.0f}-second audio segment.\n"
        f"Target classes (ID: meaning): {render_id_list(target_ids)}\n"
        "List all IDs from the targets that occur at least once in this segment.\n"
        "Output: JSON array of unique integers in ascending order. Example: [1,4] or []."
    )


def call_model_setA(audio_arr, sr, target_ids: List[int], win_len: float) -> str:
    from mistral_common.protocol.instruct.messages import TextChunk, UserMessage
    audio_chunk = array_to_audiochunk(audio_arr, sr)
    text_chunk  = TextChunk(text=build_prompt_setA(target_ids, win_len))
    user_msg    = UserMessage(content=[audio_chunk, text_chunk]).to_openai()
    payload = {
        "model": MODEL_NAME,
        "messages": [user_msg],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
    }
    r = requests.post(VLLM_ENDPOINT, json=payload, timeout=REQUEST_TIMEOUT)
    if not r.ok:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
    return r.json()["choices"][0]["message"]["content"]

# ---------------- JSON 提取/解析（优先数组） ----------------
def extract_top_level_json_value(text: str) -> Optional[str]:
    """优先从 ```json ...``` 或 ```...``` 中提取顶层 JSON；先找数组 `[...]`，再对象。"""
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    if m:
        candidate = m.group(1)
        j = _find_first_top_level(candidate, prefer_array=True)
        if j is not None:
            return j
    return _find_first_top_level(text, prefer_array=True)

def _find_first_top_level(s: str, prefer_array: bool = False) -> Optional[str]:
    pairs = [('[', ']'), ('{', '}')] if prefer_array else [('{', '}'), ('[', ']')]
    for start_char, end_char in pairs:
        start = s.find(start_char)
        while start != -1:
            i, depth, in_str, esc = start, 0, False, False
            while i < len(s):
                ch = s[i]
                if in_str:
                    if esc: esc = False
                    elif ch == '\\': esc = True
                    elif ch == '"': in_str = False
                else:
                    if ch == '"': in_str = True
                    elif ch == start_char: depth += 1
                    elif ch == end_char:
                        depth -= 1
                        if depth == 0: return s[start:i+1]
                i += 1
            start = s.find(start_char, start + 1)
    return None

def parse_setA_array(raw_reply: str, allowed_ids: Set[int]) -> Optional[List[int]]:
    """
    解析形态A：必须是整数数组；过滤非整数/越界/不在 allowed 的元素；升序去重。
    若无法解析出数组，返回 None（记为解析失败）。
    """
    j = extract_top_level_json_value(raw_reply)
    if not j:
        return None
    try:
        data = json.loads(j)
    except Exception:
        return None
    if not isinstance(data, list):
        return None
    out = []
    for x in data:
        try:
            xi = int(x)
        except Exception:
            continue
        if xi in allowed_ids:
            out.append(xi)
    if not out:
        return []
    out = sorted(set(out))
    return out

# ---------------- 指标计算 ----------------
def compute_metrics_per_win(gt_set: Set[int], pred_set: Set[int], class_ids: List[int]) -> Dict[str, float]:
    """返回该窗口的基础指标（Jaccard/Subset/Hamming等）。"""
    y_true = [1 if c in gt_set else 0 for c in class_ids]
    y_pred = [1 if c in pred_set else 0 for c in class_ids]

    # Subset Accuracy：完全匹配
    subset_acc = 1.0 if y_true == y_pred else 0.0

    # Hamming Loss：逐位错误率
    hamm = sum(int(a != b) for a, b in zip(y_true, y_pred)) / len(class_ids) if class_ids else 0.0

    # Jaccard (IoU)
    inter = len(gt_set & pred_set)
    union = len(gt_set | pred_set)
    jacc = (inter / union) if union > 0 else (1.0 if inter == 0 else 0.0)

    # Precision/Recall/F1（micro定义在整体上算，这里先返回 per-win 的 TP/FP/FN 计数用于聚合）
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)

    return {
        "subset_acc": subset_acc,
        "hamming_loss": hamm,
        "jaccard": jacc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }

def aggregate_metrics(detail_df: pd.DataFrame, class_ids: List[int]) -> pd.DataFrame:
    """
    从 detail 明细聚合：
      - micro P/R/F1（基于全局 TP/FP/FN）
      - macro P/R/F1（先算每类的PRF1再平均）
      - subset_acc / hamming_loss / jaccard 的均值
      - 每类 PRF1（供诊断）
    """
    # 按 win_len 聚合
    out_rows = []
    for win_len, g in detail_df.groupby("win_len"):
        # micro
        tp = g["tp"].sum()
        fp = g["fp"].sum()
        fn = g["fn"].sum()
        micro_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        micro_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        micro_f1 = 2*micro_p*micro_r / (micro_p+micro_r) if (micro_p+micro_r) > 0 else 0.0

        # macro（需要每类的 TP/FP/FN）
        per_class = {}
        for cid in class_ids:
            # 每类基于窗口级 0/1 标签累积
            # 这里我们需要 detail_df 里的 y_true/y_pred 向量，简化起见，直接从字符串列重建
            # （我们会在 detail_df 里保存 y_true_vec / y_pred_vec 两列 JSON 字符串）
            tp_c = fp_c = fn_c = 0
            for _, row in g.iterrows():
                ytrue = set(json.loads(row["gt_list"]))  # list of ints
                ypred = set(json.loads(row["pred_list"])) if row["pred_list"] else set()
                t = (cid in ytrue)
                p = (cid in ypred)
                if t and p: tp_c += 1
                elif (not t) and p: fp_c += 1
                elif t and (not p): fn_c += 1
            per_class[cid] = (tp_c, fp_c, fn_c)

        macro_ps, macro_rs, macro_f1s = [], [], []
        for cid, (tp_c, fp_c, fn_c) in per_class.items():
            p = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
            r = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
            f1 = 2*p*r / (p+r) if (p+r) > 0 else 0.0
            macro_ps.append(p); macro_rs.append(r); macro_f1s.append(f1)

        macro_p = float(np.mean(macro_ps)) if macro_ps else 0.0
        macro_r = float(np.mean(macro_rs)) if macro_rs else 0.0
        macro_f1 = float(np.mean(macro_f1s)) if macro_f1s else 0.0

        out_rows.append({
            "win_len": win_len,
            "micro_precision": micro_p,
            "micro_recall": micro_r,
            "micro_f1": micro_f1,
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "macro_f1": macro_f1,
            "subset_acc_mean": g["subset_acc"].mean(),
            "hamming_loss_mean": g["hamming_loss"].mean(),
            "jaccard_mean": g["jaccard"].mean(),
        })
    return pd.DataFrame(out_rows)

# ---------------- 主流程 ----------------
def ensure_dir_for(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def run_setA_for_file(wav_path: str, events_df: pd.DataFrame, target_ids: List[int]):
    base = os.path.basename(wav_path)
    base_noext = os.path.splitext(base)[0]
    allowed_ids = set(target_ids)

    for win_len in WIN_LENS:
        wav, sr, wins = split_fixed_win(wav_path, win_len)
        detail_csv    = DETAIL_CSV_TPL.format(basename=base_noext, win=win_len)
        summary_csv   = SUMMARY_CSV_TPL.format(basename=base_noext, win=win_len)
        replies_jsonl = REPLIES_JSONL_TPL.format(basename=base_noext, win=win_len)
        ensure_dir_for(replies_jsonl)

        rows = []
        with open(replies_jsonl, "a", encoding="utf-8") as fout:
            for (i0, i1, t0, t1) in wins:
                clip = wav[i0:i1]
                gt_set = present_classes_in_window(events_df, base, t0, t1, allowed_ids)

                # 调模型
                raw = call_model_setA(clip, sr, target_ids, win_len)
                fout.write(json.dumps({
                    "file": base, "win_start": t0, "win_end": t1,
                    "target_ids": target_ids, "reply": raw
                }, ensure_ascii=False) + "\n")

                # 解析预测集合
                parsed = parse_setA_array(raw, allowed_ids)  # None=解析失败，[]/list 合法
                pred_set = set(parsed) if parsed is not None else set()

                # 指标（逐窗）
                m = compute_metrics_per_win(gt_set, pred_set, target_ids)

                rows.append({
                    "file": base,
                    "win_start": t0, "win_end": t1, "win_len": win_len,
                    "gt_list": json.dumps(sorted(list(gt_set))),
                    "pred_list": json.dumps(sorted(list(pred_set))),
                    "subset_acc": m["subset_acc"],
                    "hamming_loss": m["hamming_loss"],
                    "jaccard": m["jaccard"],
                    "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
                    "raw_reply": raw
                })

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(detail_csv, index=False)

            # 聚合指标
            summary = aggregate_metrics(df, target_ids)
            summary.to_csv(summary_csv, index=False)

            print(f"[detail] {len(df)} rows -> {detail_csv}")
            print(f"[summary] {len(summary)} rows -> {summary_csv}")
        else:
            print(f"[WARN] no windows for {base} @ {win_len}s")

# ---------------- 入口 ----------------
if __name__ == "__main__":
    # 载入真实事件
    events_df = load_events_csvs(EVENTS_CSVS)

    # 跑每个文件
    for wav_path in FILES:
        if not os.path.exists(wav_path):
            print(f"[ERR] not found: {wav_path}")
            continue
        run_setA_for_file(wav_path, events_df, TARGET_CLASS_SET)

    print("\nDone. Inspect *.Aset.detail.csv and *.Aset.summary.csv\n")
