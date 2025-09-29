#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Closed-set Multi-label Detection with Qwen2-Audio (Form A: set output)

流程与 Voxtral 版一致：
- 切窗 -> 真实存在性集合 -> 询问模型（限定在给定类集合 S 内） -> 模型返回 [id,...]（升序、去重；无则 []）-> 解析 -> 指标

需要的真实标注 CSV 列：
  file,onset,offset,class
  其中 file 与音频 basename 匹配；onset/offset 单位为秒；class 为整型类别ID
"""

import os
import re
import json
import math
import warnings
import soundfile as sf
import pandas as pd
import numpy as np
from typing import List, Dict, Set, Tuple, Optional

# =============== Qwen2-Audio 模型与推理配置（按需修改） ===============
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import torch
import librosa

# 本地或远程模型标识：二选一
QWEN_MODEL_NAME_OR_PATH = "/data/user/jzt/.cache/modelscope/hub/models/Qwen/Qwen2-Audio-7B-Instruct"
# 也可用 HuggingFace Hub 名称：
# QWEN_MODEL_NAME_OR_PATH = "Qwen/Qwen2-Audio-7B-Instruct"

QWEN_LOCAL_ONLY   = True   # 仅本地加载
QWEN_DEVICE_MAP   = "auto"
QWEN_MAX_NEW_TOKENS = 4096
QWEN_TEMPERATURE  = 0.2    # 设为 0 更可复现；需要采样可改 >0
QWEN_TOP_P        = 0.95

# =============== 评测配置（与 Voxtral 版保持一致） ===============
# 窗口长度（秒）
WIN_LENS = [5,10,20,30,40,50,60]

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
TARGET_CLASS_SET = sorted([0, 1, 2, 4])
# 或者用全类：
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

# 输出模板（与 Voxtral 版一一对应）
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
    return wav.astype(np.float32, copy=False), sr, wins

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
    # 有交叠即视为“出现过”
    mask = (sub["offset"] > t0) & (sub["onset"] < t1)
    cats = set(int(c) for c in sub.loc[mask, "class"].tolist())
    return set(c for c in cats if c in target_set)

# ---------------- 提示词 & 模型调用（形态 A：集合输出） ----------------
def render_id_list(target_ids: List[int]) -> str:
    parts = [f"{cid}: {CLASS_ID_TO_NAME.get(cid,'Unknown')}" for cid in target_ids]
    return "[" + ", ".join(parts) + "]"

def build_prompt_setA(target_ids: List[int], win_len: float) -> str:
    # 与 Voxtral 版保持等价语义与约束
    return (
        f"This is a {win_len:.0f}-second audio segment.\n"
        f"Target classes (ID: meaning): {render_id_list(target_ids)}\n"
        "List all IDs from the targets that occur at least once in this segment.\n"
        "Output: JSON array of unique integers in ascending order. Example: [1,4] or []."
    )

# ---- Qwen 推理：用 Qwen2-AudioForConditionalGeneration + AutoProcessor ----
_qwen_processor = None
_qwen_model = None
_QWEN_TARGET_SR = None

def _lazy_load_qwen():
    global _qwen_processor, _qwen_model, _QWEN_TARGET_SR
    if _qwen_processor is None or _qwen_model is None:
        _qwen_processor = AutoProcessor.from_pretrained(
            QWEN_MODEL_NAME_OR_PATH,
            sampling_rate=16000,
            local_files_only=QWEN_LOCAL_ONLY
        )
        _qwen_model = Qwen2AudioForConditionalGeneration.from_pretrained(
            QWEN_MODEL_NAME_OR_PATH,
            local_files_only=QWEN_LOCAL_ONLY,
            device_map=QWEN_DEVICE_MAP
        )
        _QWEN_TARGET_SR = _qwen_processor.feature_extractor.sampling_rate

def _resample_to_qwen(audio_arr: np.ndarray, sr: int) -> np.ndarray:
    if sr == _QWEN_TARGET_SR:
        return np.asarray(audio_arr, dtype=np.float32, order="C")
    y = librosa.resample(audio_arr.astype(np.float32, copy=False), orig_sr=sr, target_sr=_QWEN_TARGET_SR, res_type="kaiser_fast")
    return np.ascontiguousarray(y, dtype=np.float32)

def call_model_setA_qwen(audio_arr: np.ndarray, sr: int, target_ids: List[int], win_len: float) -> str:
    """
    与 Voxtral 版 call_model_setA 等价的功能：返回文本（模型回复）。
    这里采用 Qwen 官方推荐用法：processor.apply_chat_template + audio 张量一起喂给模型。
    为了鲁棒解析，不强制裁剪“新生成部分”，直接整体 decode。
    """
    _lazy_load_qwen()

    # 采样率处理
    audio16k = _resample_to_qwen(audio_arr, sr)

    # 构造会话
    prompt = build_prompt_setA(target_ids, win_len)
    conversation = [
        {"role": "user", "content": [
            {"type": "audio", "audio_url": None},  # 本地数组走 processor 的 audio 参数；占位无实际 URL
            {"type": "text", "text": prompt},
        ]},
    ]
    # ChatML 文本
    chat_text = _qwen_processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

    # 打包张量
    inputs = _qwen_processor(text=chat_text, audio=audio16k, return_tensors="pt", padding=True)
    # 移到设备
    inputs = {k: v.to(_qwen_model.device) for k, v in inputs.items()}

    # 生成
    with torch.no_grad():
        outputs = _qwen_model.generate(
            **inputs,
            max_new_tokens=QWEN_MAX_NEW_TOKENS,
            do_sample=(QWEN_TEMPERATURE > 0.0),
            temperature=QWEN_TEMPERATURE,
            top_p=QWEN_TOP_P,
        )
    input_ids_length = inputs['input_ids'].size(1)
    outputs = outputs[:, input_ids_length:]
    # 直接 decode 全部，然后交给解析器抽 JSON
    text = _qwen_processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return text

# ---------------- JSON 提取/解析（优先数组，等价 Voxtral 逻辑） ----------------
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

def parse_setA_array(raw_reply: str, allowed_ids: Set[int]) -> Tuple[bool, Optional[List[int]]]:
    """
    解析形态A：必须是整数数组；过滤非整数/越界/不在 allowed 的元素；升序去重。
    返回 (parse_ok, result)
      - parse_ok = True 且 result 为 list：解析成功（可为空列表）
      - parse_ok = False：解析失败（result = None）
    """
    j = extract_top_level_json_value(raw_reply)
    if not j:
        return (False, None)
    try:
        data = json.loads(j)
    except Exception:
        return (False, None)
    if not isinstance(data, list):
        return (False, None)
    out = []
    for x in data:
        try:
            xi = int(x)
        except Exception:
            continue
        if xi in allowed_ids:
            out.append(xi)
    out = sorted(set(out))
    return (True, out)  # 允许空数组

# ---------------- 指标计算（与 Voxtral 版一致） ----------------
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
    """
    out_rows = []
    for win_len, g in detail_df.groupby("win_len"):
        # micro
        tp = g["tp"].sum()
        fp = g["fp"].sum()
        fn = g["fn"].sum()
        micro_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        micro_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        micro_f1 = 2*micro_p*micro_r / (micro_p+micro_r) if (micro_p+micro_r) > 0 else 0.0

        # macro（逐类窗口级计数）
        per_class = {}
        for cid in class_ids:
            tp_c = fp_c = fn_c = 0
            for _, row in g.iterrows():
                ytrue = set(json.loads(row["gt_list"]))
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
            # 解析成功率（便于区分“空预测”与“解析失败”）
            "parse_ok_rate": g["parse_ok"].mean(),
        })
    return pd.DataFrame(out_rows)

# ---------------- 主流程 ----------------
def ensure_dir_for(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def run_setA_for_file_qwen(wav_path: str, events_df: pd.DataFrame, target_ids: List[int]):
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
        # 每次运行建议覆盖写；如需历史追加，将 "w" 改回 "a"
        with open(replies_jsonl, "w", encoding="utf-8") as fout:
            for (i0, i1, t0, t1) in wins:
                clip = wav[i0:i1]
                gt_set = present_classes_in_window(events_df, base, t0, t1, allowed_ids)

                # 调模型（Qwen）
                try:
                    raw = call_model_setA_qwen(clip, sr, target_ids, win_len)
                except Exception as e:
                    raw = f"[ERROR] {type(e).__name__}: {e}"

                # 审计原始回复
                fout.write(json.dumps({
                    "file": base, "win_start": t0, "win_end": t1,
                    "target_ids": target_ids, "reply": raw
                }, ensure_ascii=False) + "\n")

                # 解析预测集合（与 Voxtral 版一致的规则 + parse_ok 区分）
                parse_ok, parsed = parse_setA_array(raw, allowed_ids)
                pred_set = set(parsed) if (parse_ok and parsed is not None) else set()

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
                    "parse_ok": 1 if parse_ok else 0,   # 新增：解析成功标记
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
    # 固定随机性（可选）
    torch.manual_seed(0)

    # 载入真实事件
    events_df = load_events_csvs(EVENTS_CSVS)

    # 跑每个文件
    for wav_path in FILES:
        if not os.path.exists(wav_path):
            print(f"[ERR] not found: {wav_path}")
            continue
        run_setA_for_file_qwen(wav_path, events_df, TARGET_CLASS_SET)

    print("\nDone. Inspect *.Aset.detail.csv and *.Aset.summary.csv\n")
