#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用 Qwen2-Audio-7B-Instruct 复现“Voxtral 起点实验”（每条样本询问 3 次取均值）。

- 输入：manifest CSV（含 new_out_path,class,event_onset_sec,event_offset_sec,window_sec,...）。
- 对每行音频：用极简提示词 + 音频输入，要求模型仅输出 {"start time": <number>}（0..W, 保留 1 位小数）。
- 每条样本调用模型 3 次，解析成功的值做均值（1 位小数）。
- 输出：结果 CSV，保留元信息 + pred_onset_sec + 3 次原始回复拼接。

依赖：
  pip install -U transformers torch soundfile librosa pandas numpy
  # 建议准备 GPU 环境；权重可指向本地目录（QWEN_LOCAL_DIR）
"""

import os
import re
import json
import argparse
from typing import Optional, Tuple, Any, List

import numpy as np
import pandas as pd
import librosa
import torch
from transformers import Qwen2AudioForConditionalGeneration, Qwen2AudioProcessor

# ------- 类别到名称（可按需修改/扩展） -------
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

JSON_PATTERN = re.compile(r"\{.*?\}", re.DOTALL)

# -------------------- QWEN 全局（延迟加载） --------------------

_qwen_model = None
_qwen_processor = None
_QWEN_TARGET_SR = 16000


def load_qwen(model_dir: str, device: str = "cuda:1", dtype: str = "float16"):
    global _qwen_model, _qwen_processor, _QWEN_TARGET_SR
    if _qwen_model is not None:
        return
    _qwen_processor = Qwen2AudioProcessor.from_pretrained(model_dir, sampling_rate=16000)
    _qwen_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_dir,
        device_map={"": 1} if device.startswith("cuda") else None,
        torch_dtype=getattr(torch, dtype)
    ).eval()
    _QWEN_TARGET_SR = getattr(_qwen_processor, "feature_extractor", None).sampling_rate if hasattr(_qwen_processor, "feature_extractor") else 16000


# -------------------- Prompt --------------------

def build_prompt_qwen(cls_id: int, window_sec: float) -> str:
    cls_name = CLASS_ID_TO_NAME.get(int(cls_id), "Unknown")
    W = float(window_sec)
    # 与 Voxtral 版本对齐：单类、要求严格 JSON
    prompt = (
        f"You will receive a single {W:.0f}-second audio clip that contains an audio event.\n"
        f"The audio event is (\"{cls_name}\").\n\n"
        "Return ONLY the following strict JSON (no extra text, no markdown, no code fences):\n"
        "{\"start time\": <number>}\n\n"
        f"Constraints:\n- 0 <= start time <= {W:.0f}\n- Use seconds with up to 1 decimal place (e.g., 3.1).\n"
        "The \"start time\" must be RELATIVE to THIS clip (0 means the clip start)."
    )
    return prompt


# -------------------- 解析 --------------------

def parse_json_from_text(text: str) -> Optional[dict]:
    if not text:
        return None
    # 直接尝试
    try:
        return json.loads(text)
    except Exception:
        pass
    # 正则提取第一个 JSON 对象
    m = JSON_PATTERN.search(text)
    if m:
        try:
            return json.loads(m.group(0).replace("'", '"'))
        except Exception:
            return None
    return None


def _to_valid_time(x, window_sec: float) -> Optional[float]:
    try:
        f = float(x)
        if 0.0 <= f <= float(window_sec) + 1e-9:
            return round(f + 1e-9, 1)
        return None
    except Exception:
        return None


def extract_start_time(data: Any, window_sec: float) -> Optional[float]:
    if not isinstance(data, dict):
        return None
    # 优先严格键名
    if "start time" in data:
        return _to_valid_time(data["start time"], window_sec)
    # 容错键名
    for k in ["start_time", "start", "onset", "pred_onset_sec"]:
        if k in data:
            return _to_valid_time(data[k], window_sec)
    # 宽松：找首个可解析数值
    for v in data.values():
        cand = _to_valid_time(v, window_sec)
        if cand is not None:
            return cand
    return None


# -------------------- QWEN 调用（一次） --------------------

def qwen_predict_once(audio_path: str,
                      cls_id: int,
                      window_sec: float,
                      do_sample: bool,
                      temperature: float,
                      top_p: float,
                      max_new_tokens: int) -> Tuple[Optional[float], str]:
    """对同一音频**一次**调用 Qwen，解析 {"start time":<number>} 并返回值与原始文本。"""
    # 加载音频到目标采样率，转换为单通道
    wav, _ = librosa.load(audio_path, sr=_QWEN_TARGET_SR, mono=True)
    # 构造 ChatML 对话（音频 + 文本）
    prompt = build_prompt_qwen(cls_id, window_sec)
    conversation = [
        {"role": "user", "content": [
            {"type": "audio", "audio_url": None},
            {"type": "text", "text": prompt},
        ]}
    ]
    text = _qwen_processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = _qwen_processor(text=text, audio=wav, return_tensors="pt", padding=True)
    # 移动到设备
    inputs = {k: (v.to(_qwen_model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    # 生成
    with torch.no_grad():
        gen_ids = _qwen_model.generate(
            **inputs,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
    # 只取新生成部分
    gen_ids = gen_ids[:, inputs["input_ids"].size(1):]
    reply = _qwen_processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    data = parse_json_from_text(reply) or {}
    start_time = extract_start_time(data, window_sec)
    return start_time, reply


# -------------------- 主流程 --------------------

def main():
    parser = argparse.ArgumentParser(description="Qwen2-Audio onset-only, ask 3 times and average valid results.")
    parser.add_argument("--manifest", required=True, help="输入清单 CSV（含 new_out_path,class,event_onset_sec,event_offset_sec,window_sec,...）")
    parser.add_argument("--out", required=True, help="输出结果 CSV 路径")

    # Qwen 模型设置
    parser.add_argument("--qwen-local-dir", required=True, help="Qwen2-Audio-7B-Instruct 本地权重目录")
    parser.add_argument("--device", default="cuda:0", help="设备，如 cuda:0 / cpu")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"], help="模型精度")

    # 生成参数
    parser.add_argument("--do-sample", action="store_true", help="开启抽样（默认关闭以更稳定 JSON）")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=1024)

    # 列名
    parser.add_argument("--col-audio", default="new_out_path")
    parser.add_argument("--col-class", default="class")
    parser.add_argument("--col-gt-onset", default="event_onset_sec")
    parser.add_argument("--col-gt-offset", default="event_offset_sec")
    parser.add_argument("--col-window", default="window_sec")
    parser.add_argument("--col-seg-idx", default="segment_index")
    parser.add_argument("--col-placement", default="placement")
    parser.add_argument("--col-source", default="source")

    args = parser.parse_args()

    # 加载 Qwen
    load_qwen(args.qwen_local_dir, device=args.device, dtype=args.dtype)

    df = pd.read_csv(args.manifest)
    need_cols = {args.col_audio, args.col_class, args.col_gt_onset, args.col_gt_offset, args.col_window}
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"清单缺少列：{missing}\n当前列：{list(df.columns)}")

    results = []
    for i, row in df.iterrows():
        audio_path = str(row[args.col_audio])
        cls_id     = int(row[args.col_class])
        gt_onset   = float(row[args.col_gt_onset])
        gt_offset  = float(row[args.col_gt_offset])
        window_sec = float(row[args.col_window])

        seg_idx   = row[args.col_seg_idx] if args.col_seg_idx in df.columns else None
        placement = row[args.col_placement] if args.col_placement in df.columns else None
        source    = row[args.col_source] if args.col_source in df.columns else None

        if not os.path.exists(audio_path):
            print(f"[SKIP {i}] 文件不存在：{audio_path}")
            results.append({
                "audio_path": audio_path,
                "class": cls_id,
                "class_name": CLASS_ID_TO_NAME.get(cls_id, "Unknown"),
                "gt_onset_sec": round(gt_onset, 1),
                "gt_offset_sec": round(gt_offset, 1),
                "window_sec": int(window_sec),
                "segment_index": seg_idx,
                "placement": placement,
                "source": source,
                "pred_onset_sec": None,
                "raw_model_output": "[file_not_found]"
            })
            continue

        print(f"[{i+1}/{len(df)}] Qwen 推理: {audio_path} (class={cls_id}, W={window_sec:.0f}s, seg={seg_idx}, place={placement})")

        # 三次独立调用
        vals: List[Optional[float]] = []
        raws: List[str] = []
        for t in range(3):
            v, r = qwen_predict_once(
                audio_path=audio_path,
                cls_id=cls_id,
                window_sec=window_sec,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
            )
            vals.append(v)
            raws.append(r)

        valid = [x for x in vals if x is not None]
        pred_mean = round(float(np.mean(valid)), 1) if valid else None

        results.append({
            "audio_path": audio_path,
            "class": cls_id,
            "class_name": CLASS_ID_TO_NAME.get(cls_id, "Unknown"),
            "gt_onset_sec": round(gt_onset, 1),
            "gt_offset_sec": round(gt_offset, 1),
            "window_sec": int(window_sec),
            "segment_index": seg_idx,
            "placement": placement,
            "source": source,
            "pred_onset_sec": pred_mean,
            "raw_model_output": " ||| ".join(raws),
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.out, index=False)
    print(f"[DONE] 写出结果：{args.out}，共 {len(out_df)} 条")


if __name__ == "__main__":
    main()
