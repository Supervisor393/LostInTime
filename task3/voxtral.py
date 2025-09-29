#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
按窗口长度（10/20/30/40/50/60s）调用 vLLM(OpenAI 兼容)上的 Voxtral，
使用简洁提示词让模型输出 {"start time": <number>}（1 位小数），
并将每个音频片段**询问 3 次、取均值**作为预测结果，保存结果 CSV。

变更点：
- 取消原先的重试机制；
- 对每条样本进行 3 次独立调用，分别解析数值后求平均（只对成功解析且通过范围校验的值求均值）；
- 其他逻辑保持不变（严格 JSON 抽取与范围校验、保留元信息、写出 CSV 等）。

依赖：
  pip install pandas openai mistral-common huggingface_hub
"""

import os
import re
import json
import argparse
import pandas as pd
from typing import Optional, Tuple, Any, List

from openai import OpenAI
from mistral_common.protocol.instruct.messages import TextChunk, AudioChunk, UserMessage
from mistral_common.audio import Audio

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

JSON_PATTERN = re.compile(r"\{.*?\}", re.DOTALL)  # 非贪婪，避免跨多段

# -------------------- Prompt --------------------

def file_to_chunk(path: str) -> AudioChunk:
    audio = Audio.from_file(path, strict=False)
    return AudioChunk.from_audio(audio)

def build_user_message(audio_path: str, cls_id: int, window_sec: float) -> dict:
    """
    极简+严格JSON的提示词：只要 "start time"，1 位小数；窗口长度按行自适应。
    """
    audio_chunk = file_to_chunk(audio_path)
    cls_name = CLASS_ID_TO_NAME.get(int(cls_id), "Unknown")
    W = float(window_sec)

    instruction = f"""
You will receive a single {W:.0f}-second audio clip that contains an audio event. 
And you need to give the start time of it. The audio event is ("{cls_name}").

Return ONLY the following strict JSON (no extra text, no code fences):
{{"start time": <number>}}

Constraints:
- 0 <= start time <= {W:.0f}
- Use seconds with up to 1 decimal places (e.g., 3.1)
""".strip()

    text_chunk = TextChunk(text=instruction)
    user_msg = UserMessage(content=[audio_chunk, text_chunk]).to_openai()
    return user_msg

# -------------------- Parsing helpers --------------------

def parse_json_from_text(text: str) -> Optional[dict]:
    # 直接尝试
    try:
        return json.loads(text)
    except Exception:
        pass
    # 容错：提取首个 JSON 块
    m = JSON_PATTERN.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None

def extract_start_time(data: Any, window_sec: float) -> Optional[float]:
    """
    从返回的 dict 中拿到 "start time"（兼容一些常见误差键名），
    并做范围与保留 1 位小数处理。
    """
    if not isinstance(data, dict):
        return None

    # 优先严格键名
    if "start time" in data:
        return _to_valid_time(data["start time"], window_sec)

    # 容错键名
    for k in ["start_time", "start", "onset", "pred_onset_sec"]:
        if k in data:
            return _to_valid_time(data[k], window_sec)

    # 再宽松一些：找首个数值
    for v in data.values():
        cand = _to_valid_time(v, window_sec)
        if cand is not None:
            return cand

    return None

def _to_valid_time(x, window_sec: float) -> Optional[float]:
    try:
        f = float(x)
        if 0.0 <= f <= float(window_sec):
            # 最终要求 1 位小数
            return round(f + 1e-9, 1)
        return None
    except Exception:
        return None

# -------------------- Inference --------------------

def ask_voxtral_once(client: OpenAI, model: str, audio_path: str, cls_id: int, window_sec: float) -> Tuple[Optional[float], str]:
    """
    对同一音频**一次**调用模型，解析 {"start time": <number>}。
    返回 (parsed_start_time or None, raw_text)
    """
    user_msg = build_user_message(audio_path, cls_id, window_sec)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[user_msg],
            temperature=0.2,
            top_p=0.95,
            max_tokens=1024,
        )
        content = resp.choices[0].message.content
    except Exception as e:
        return None, f"[error:{e}]"

    data = parse_json_from_text(content) or {}
    start_time = extract_start_time(data, window_sec)
    return start_time, content

# -------------------- Main --------------------

def main():
    parser = argparse.ArgumentParser(description="Voxtral onset-only, per-row window length, ask 3 times and average the valid results.")
    parser.add_argument("--manifest", required=True, help="输入清单 CSV（含 new_out_path,class,event_onset_sec,event_offset_sec,window_sec,...）")
    parser.add_argument("--out", required=True, help="输出结果 CSV 路径")
    parser.add_argument("--api-base", default="http://127.0.0.1:8011/v1", help="OpenAI 兼容 API base")
    parser.add_argument("--api-key", default="EMPTY", help="OpenAI API key（vLLM 用占位即可）")
    parser.add_argument("--model", default="voxtral-mini-3b", help="模型名（served-model-name）")

    # 你的清单默认列名（与示例一致）
    parser.add_argument("--col-audio", default="new_out_path", help="音频路径列名（默认 new_out_path）")
    parser.add_argument("--col-class", default="class", help="类别列名（默认 class）")
    parser.add_argument("--col-gt-onset", default="event_onset_sec", help="GT 起始时间列名（默认 event_onset_sec）")
    parser.add_argument("--col-gt-offset", default="event_offset_sec", help="GT 结束时间列名（默认 event_offset_sec）")
    parser.add_argument("--col-window", default="window_sec", help="窗口长度列名（默认 window_sec）")

    # 保留的额外元信息（若存在，则写回结果，便于分析）
    parser.add_argument("--col-seg-idx", default="segment_index", help="段索引列名（默认 segment_index）")
    parser.add_argument("--col-placement", default="placement", help="摆放标签列名（默认 placement）")
    parser.add_argument("--col-source", default="source", help="来源列名（默认 source）")

    args = parser.parse_args()

    df = pd.read_csv(args.manifest)

    # 校验列
    need_cols = {args.col_audio, args.col_class, args.col_gt_onset, args.col_gt_offset, args.col_window}
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"清单缺少列：{missing}\n当前列：{list(df.columns)}")

    client = OpenAI(api_key=args.api_key, base_url=args.api_base)

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
            print(f"[SKIP {i}] 文件不存在：\"{audio_path}\"")
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

        print(f"[{i+1}/{len(df)}] 推理: {audio_path} (class={cls_id}, W={window_sec:.0f}s, seg={seg_idx}, place={placement})")

        # 三次独立调用
        parsed_values: List[Optional[float]] = []
        raw_texts: List[str] = []
        for trial in range(3):
            val, raw = ask_voxtral_once(
                client=client,
                model=args.model,
                audio_path=audio_path,
                cls_id=cls_id,
                window_sec=window_sec,
            )
            parsed_values.append(val)
            raw_texts.append(raw)

        # 仅对成功解析且合法的值求均值
        valid_vals = [v for v in parsed_values if v is not None]
        pred_mean = round(sum(valid_vals) / len(valid_vals), 1) if valid_vals else None

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
            # 将三次原始输出拼接，便于排查；如需更结构化可改为 JSON
            "raw_model_output": " ||| ".join(raw_texts),
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.out, index=False)
    print(f"[DONE] 写出结果：{args.out}，共 {len(out_df)} 条")

if __name__ == "__main__":
    main()