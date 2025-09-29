# --- put this near the top of your script, BEFORE loading the model ---
import sys, types

def _ensure_dummy_videoinput():
    try:
        import transformers.video_utils as vu
        if not hasattr(vu, "VideoInput"):
            class VideoInput:
                def __init__(self, pixel_values=None, num_frames=None, fps=None, **kwargs):
                    self.pixel_values = pixel_values
                    self.num_frames = num_frames
                    self.fps = fps
            vu.VideoInput = VideoInput
    except Exception:
        video_utils = types.ModuleType("transformers.video_utils")
        class VideoInput:
            def __init__(self, pixel_values=None, num_frames=None, fps=None, **kwargs):
                self.pixel_values = pixel_values
                self.num_frames = num_frames
                self.fps = fps
        video_utils.VideoInput = VideoInput
        video_utils.__all__ = ["VideoInput"]
        sys.modules["transformers.video_utils"] = video_utils

_ensure_dummy_videoinput()
# --- end of patch ---

import os
import random
import json
import re
import time
import soundfile as sf
import pandas as pd
import numpy as np
import librosa
import torch
from typing import List, Dict, Any, Optional

# 代理（如无需可注释）
os.environ["http_proxy"] = "http://127.0.0.1:9090"
os.environ["https_proxy"] = "http://127.0.0.1:9090"
# 你可以在程序开头定义一个输出目录路径
OUTPUT_DIR = "/data/user/jzt/crd/audioLLM/new_data/areo"  # 请根据需要修改为实际路径

# ========== AERO MODEL CONFIG ==========
AERO_MODEL_ID = "/data/user/jzt/crd/audioLLM/Aero-1-Audio"
AERO_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {AERO_DEVICE}")
AERO_ATTN_IMPL = "flash_attention_2"   # 使用 sdpa，避免安装 flash-attn
AERO_SR = 16000

# 生成参数
TEMPERATURE = 0.2
TOP_P = 0.95
MAX_TOKENS = 256

# 窗口长度（秒）
WIN_LENS = [5,10,20,30,40,50,60,90,120]

TAU_EVENTS_CSV  = "/data/user/jzt/crd/audioLLM/train_events/tau.csv"
SONY_EVENTS_CSV = "/data/user/jzt/crd/audioLLM/train_events/sony.csv"

# ====== CLASS ID → NAME MAPPING ======
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

# ---------------- Robust JSON -> list[{"class":int, "start":float}] ----------------
def extract_json_array(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    if m:
        candidate = m.group(1)
        arr = _find_top_level_array(candidate)
        if arr is not None:
            return arr
    arr = _find_top_level_array(text)
    return arr

def _find_top_level_array(s: str) -> Optional[str]:
    start = s.find('[')
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
                elif ch == '[': depth += 1
                elif ch == ']':
                    depth -= 1
                    if depth == 0:
                        return s[start:i+1]
            i += 1
        start = s.find('[', start + 1)
    return None

def parse_class_start_list(raw_reply: str, win_len: float) -> List[Dict[str, Any]]:
    arr_text = extract_json_array(raw_reply)
    if not arr_text:
        return []
    try:
        data = json.loads(arr_text)
    except Exception:
        return []
    out = []
    if isinstance(data, list):
        for e in data:
            if not isinstance(e, dict):
                continue
            c = e.get("class"); st = e.get("start")
            try:
                c_int = int(c); st_f = float(st)
            except Exception:
                continue
            if 0 <= c_int <= 12 and (0.0 <= st_f < win_len + 1e-6):
                out.append({"class": c_int, "start": st_f})
    return out

# ---------------- audio utils ----------------
def ensure_mono(wav: np.ndarray) -> np.ndarray:
    if hasattr(wav, "ndim") and wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    return wav.astype(np.float32)

def resample_to_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    if sr == AERO_SR:
        return wav
    return librosa.resample(wav, orig_sr=sr, target_sr=AERO_SR)

# ---------------- prompt ----------------
def build_prompt(present_classes: List[int], win_len: float) -> str:
    cls_list_str = "[" + ", ".join(
        f"{cid}: {CLASS_ID_TO_NAME.get(cid, 'Unknown')}"
        for cid in sorted(set(present_classes))
    ) + "]"
    prompt = (
        f"You will be given a {win_len:.0f}-second audio segment.\n"
        f"The following target class IDs are present in THIS segment (ID: meaning): {cls_list_str}.\n"
        "Task: For EACH listed class ID, report the EARLIEST start time within THIS segment.\n"
        "Return ONLY a strict JSON array with EXACTLY one object per listed class, sorted by class id.\n"
        "{\"class\": ID, \"start\": seconds} for each object.\n"
        f"Constraints: 0 <= start < {win_len:.1f}. 'start' must be RELATIVE to THIS segment (0 means the segment start).\n"
        "Do NOT include any class not in the list. No extra text. No markdown. No code fences."
    )
    return prompt

# ---------------- Aero model (lazy load, singletons) ----------------
_processor = None
_model = None
_eos_id = None

def _load_aero_once():
    global _processor, _model, _eos_id
    if _processor is None or _model is None:
        from transformers import AutoProcessor, AutoModelForCausalLM
        _processor = AutoProcessor.from_pretrained(AERO_MODEL_ID, trust_remote_code=True)
        _model = AutoModelForCausalLM.from_pretrained(
            AERO_MODEL_ID,
            device_map=AERO_DEVICE,
            torch_dtype="auto",
            attn_implementation=AERO_ATTN_IMPL,
            trust_remote_code=True,
        )
        _model.eval()
        _eos_id = (
            getattr(_model.generation_config, "eos_token_id", None)
            or getattr(getattr(_processor, "tokenizer", None), "eos_token_id", None)
            or 151645
        )

def call_model_identify_classes(audio_arr: np.ndarray, sr: int, win_len: float, present_classes: List[int]) -> str:
    """
    第一次提问：让模型识别音频窗口中的事件类。

    参数：
    - audio_arr: 当前窗口的音频数据（ndarray 格式）
    - sr: 音频的采样率
    - win_len: 当前窗口的长度（秒）
    - present_classes: 当前窗口内存在的真实事件类别（类 ID 列表）

    返回值：
    - 一个字符串，包含模型识别到的音频事件类别的 ID 列表
    """
    _load_aero_once()  # 确保模型已加载

    prompt = f"Please identify the types of audio events present in this {win_len}-second audio segment.\n"
    prompt += "Return a list of class IDs that correspond to the events detected in the segment.\n"
    prompt += "The class IDs should be from the following list:\n"
    prompt += "[0: Female speech, woman speaking, 1: Male speech, man speaking, 2: Clapping, 3: Telephone, 4: Laughter, 5: Domestic sounds, 6: Walk, footsteps, 7: Door, open or close, 8: Music, 9: Musical instrument, 10: Water tap, faucet, 11: Bell, 12: Knock]\n"
    prompt += "Return only the list of class IDs in a strict JSON array. Do not include any extra text."

    # 将音频处理成合适的格式
    audio_arr = ensure_mono(audio_arr)  # 确保音频是单声道
    audio_16k = resample_to_16k(audio_arr, sr)  # 重采样到 16kHz

    # 创建消息并准备模型输入
    messages = [{"role": "user", "content": [{"type": "audio_url", "audio": "placeholder"}, {"type": "text", "text": prompt}]}]
    chat_text = _processor.apply_chat_template(messages, add_generation_prompt=True)
    
    with torch.no_grad():
        # 获取模型的输入
        inputs = _processor(
            text=chat_text,
            audios=[audio_16k],
            sampling_rate=AERO_SR,
            return_tensors="pt",
        )
        inputs = {k: v.to(AERO_DEVICE) for k, v in inputs.items()}

        # 设置生成参数
        gen_kwargs = dict(
            eos_token_id=_eos_id,
            max_new_tokens=MAX_TOKENS,
            do_sample=True if TEMPERATURE > 0 else False,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )

        # 生成模型输出
        outputs = _model.generate(**inputs, **gen_kwargs)
        cont = outputs[:, inputs["input_ids"].shape[-1]:]
        text = _processor.batch_decode(cont, skip_special_tokens=True)[0]
        
        # 返回模型的响应
        return text

def call_model_get_event_times_areo(audio_arr: np.ndarray, sr: int, present_classes: List[int], win_len: float, num_retries: int = 2) -> str:
    """
    获取音频窗口的事件类别的最早开始时间，运行多次请求并计算每个类别的平均开始时间。

    参数:
    - audio_arr: 当前窗口的音频数据（ndarray 格式）
    - sr: 音频的采样率
    - present_classes: 当前窗口内出现的事件类别（类 ID 列表）
    - win_len: 当前窗口的长度（秒）
    - num_retries: 请求模型时的重试次数（默认为 2）

    返回值:
    - 模型的响应：包含每个类别的平均开始时间的字符串
    """
    all_responses = []

    # 重试 num_retries 次
    for _ in range(num_retries):
        prompt = build_prompt(present_classes, win_len)  # 构建提示词
        messages = [{"role": "user", "content": [{"type": "audio_url", "audio": "placeholder"}, {"type": "text", "text": prompt}]}]

        # 确保音频是单声道并重采样到 16kHz
        audio_arr = ensure_mono(audio_arr)
        audio_16k = resample_to_16k(audio_arr, sr)

        # 将消息转化为可供模型使用的输入格式
        chat_text = _processor.apply_chat_template(messages, add_generation_prompt=True)

        with torch.no_grad():
            # 获取模型的输入
            inputs = _processor(
                text=chat_text,
                audios=[audio_16k],
                sampling_rate=AERO_SR,
                return_tensors="pt",
            )
            inputs = {k: v.to(AERO_DEVICE) for k, v in inputs.items()}

            # 设置生成参数
            gen_kwargs = dict(
                eos_token_id=_eos_id,
                max_new_tokens=MAX_TOKENS,
                do_sample=True if TEMPERATURE > 0 else False,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )

            # 生成模型输出
            outputs = _model.generate(**inputs, **gen_kwargs)
            cont = outputs[:, inputs["input_ids"].shape[-1]:]
            text = _processor.batch_decode(cont, skip_special_tokens=True)[0]
            
            all_responses.append(text)

    # 现在我们有了 `num_retries` 次响应，将其平均处理
    return average_responses(all_responses, present_classes, win_len)


def average_responses(responses: List[str], present_classes: List[int], win_len: float) -> str:
    """
    对所有的响应进行处理，求每个类别的平均开始时间。

    参数：
    - responses: 模型多次响应的列表
    - present_classes: 当前窗口内的类别列表
    - win_len: 当前窗口的长度（秒）

    返回值：
    - 一个字符串，包含每个类别的平均开始时间
    """
    avg_results = {}

    # 收集每次响应中的类别ID及其相对开始时间
    for response in responses:
        pred_list = parse_class_start_list(response, win_len)  # 解析出每个类别的开始时间
        pred_rel_map = {int(d["class"]): float(d["start"]) for d in pred_list}

        for cid in present_classes:
            if cid in pred_rel_map:
                if cid not in avg_results:
                    avg_results[cid] = []
                avg_results[cid].append(pred_rel_map[cid])

    # 对每个类别计算平均值
    final_avg_map = {}
    for cid, times in avg_results.items():
        final_avg_map[cid] = sum(times) / len(times)  # 计算平均值

    # 构建最终的 JSON 字符串，返回给模型
    avg_response = [{"class": cid, "start": final_avg_map[cid]} for cid in sorted(final_avg_map)]
    return json.dumps(avg_response)

# ---------------- windowing and GT computation ----------------
def split_fixed_win(wav_path: str, win_len: float):
    wav, sr = sf.read(wav_path)
    wav = ensure_mono(wav)
    n = len(wav)
    samples = int(win_len * sr)
    wins = []
    i = 0
    while i + samples <= n:
        i0, i1 = i, i + samples
        t0, t1 = i0 / sr, i1 / sr
        wins.append((i0, i1, t0, t1))
        i += samples
    return wav, sr, wins

def load_events_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"file", "onset", "offset", "class"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Events CSV missing columns: {missing} in {path}")
    return df

def is_sony_path(p: str) -> bool:
    return "sony" in p.lower()

def is_tau_path(p: str) -> bool:
    return "tau" in p.lower()

def earliest_class_abs_time_in_window(df_events: pd.DataFrame, basename: str, t0: float, t1: float) -> Dict[int, float]:
    df = df_events[df_events["file"] == basename]
    if df.empty:
        return {}
    mask = (df["offset"] > t0) & (df["onset"] < t1)
    sub = df.loc[mask, ["class", "onset", "offset"]]
    if sub.empty:
        return {}
    out: Dict[int, float] = {}
    for cls_id, grp in sub.groupby("class"):
        try:
            cid = int(cls_id)
        except Exception:
            continue
        earliest_abs = min(max(float(row.onset), t0) for _, row in grp.iterrows())
        out[cid] = float(earliest_abs)
    return out

# ---------------- append writers (no dedupe) ----------------
def append_rows(csv_path: str, rows: List[Dict[str, Any]]):
    if not rows:
        return
    
    # 使用 OUTPUT_DIR 和文件名拼接得到完整路径
    output_path = os.path.join(OUTPUT_DIR, csv_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    cols = ["file", "win_start", "win_end", "class", "gt_start", "pred_start", "is_recognized"]
    df = pd.DataFrame(rows)[cols]
    exists = os.path.exists(output_path)
    df.to_csv(output_path, mode=("a" if exists else "w"), header=not exists, index=False)
    print(f"[append] {len(df)} rows -> {output_path}")


# ---------------- main per-file process ----------------
def extract_classes_from_response(response: str) -> List[int]:
    """
    提取模型回复中的类ID，假设类ID是数字，并且在响应中以一定的格式出现。
    通过正则表达式提取所有的数字。
    """
    # 正则表达式提取所有数字，假设模型返回的类ID是数字
    class_ids = re.findall(r'\b\d+\b', response)  # 匹配所有的数字

    # 将提取到的数字转换为整数列表
    class_ids = [int(cid) for cid in class_ids]

    # 去除不在合法范围内的类别ID
    class_ids = [cid for cid in class_ids if 0 <= cid <= 12]  # 只保留合法的类别ID

    return class_ids

def process_file_windows_areo(wav_path: str, events_df: pd.DataFrame, out_csv_path: str, replies_jsonl_path: str, win_len: float):
    wav, sr, wins = split_fixed_win(wav_path, win_len)
    base = os.path.basename(wav_path)
    
    # 拼接完整的文件路径
    out_csv_path = os.path.join(OUTPUT_DIR, out_csv_path)
    replies_jsonl_path = os.path.join(OUTPUT_DIR, replies_jsonl_path)

    os.makedirs(os.path.dirname(replies_jsonl_path) or ".", exist_ok=True)
    append_buffer: List[Dict[str, Any]] = []

    with open(replies_jsonl_path, "a", encoding="utf-8") as fout:
        for (i0, i1, t0, t1) in wins:
            gt_abs_map = earliest_class_abs_time_in_window(events_df, base, t0, t1)
            if not gt_abs_map:
                continue

            clip = wav[i0:i1]
            present_classes = sorted(gt_abs_map.keys())
            if len(present_classes) > 4:
                present_classes = random.sample(present_classes, 4)

            # 第一次提问：让 Areo 模型识别音频窗口中的事件类
            raw_classes_response = call_model_identify_classes(clip, sr, present_classes, win_len)
            detected_classes = extract_classes_from_response(raw_classes_response)
           
            # 第二次提问：根据真实标签中的事件类别来提问
            raw_time_response = call_model_get_event_times_areo(clip, sr, present_classes, win_len)

            # 保存原始回复（按窗口）
            fout.write(json.dumps({
                "file": base,
                "win_start": t0,
                "win_end": t1,
                "classes": present_classes,
                "reply": raw_time_response
            }, ensure_ascii=False) + "\n")

            # 解析模型相对开始时间（带 win_len 约束）
            pred_list = parse_class_start_list(raw_time_response, win_len)  # [{"class":ID,"start":rel}, ...]
            pred_rel_map = {int(d["class"]): float(d["start"]) for d in pred_list}

            # 计算时间偏差，分别记录识别和未识别的事件
            for cid in present_classes:
                gt_abs = float(gt_abs_map[cid])  # 绝对时间
                if cid in detected_classes:  # 判断是否模型识别了该事件
                    if cid in pred_rel_map:
                        pred_abs = t0 + pred_rel_map[cid]  # 模型预测的绝对时间
                        append_buffer.append({
                            "file": base,
                            "win_start": t0,
                            "win_end": t1,
                            "class": int(cid),
                            "gt_start": gt_abs,     # 绝对时间
                            "pred_start": pred_abs,  # 绝对时间（= 窗口起点 + 相对时间）
                            "is_recognized": True   # 标记为模型识别的事件
                        })
                    else:
                        append_buffer.append({
                            "file": base,
                            "win_start": t0,
                            "win_end": t1,
                            "class": int(cid),
                            "gt_start": gt_abs,     # 绝对时间
                            "pred_start": None,  # 未识别的事件，记录为空
                            "is_recognized": True   # 标记为模型识别的事件
                        })
                else:  # 模型没有识别出该事件
                    # 如果模型给出了预测的开始时间（即在 pred_rel_map 中有该类别），记录预测时间
                    if cid in pred_rel_map:
                        pred_abs = t0 + pred_rel_map[cid]  # 模型预测的绝对时间
                        append_buffer.append({
                            "file": base,
                            "win_start": t0,
                            "win_end": t1,
                            "class": int(cid),
                            "gt_start": gt_abs,     # 绝对时间
                            "pred_start": pred_abs,  # 绝对时间（= 窗口起点 + 相对时间）
                            "is_recognized": False  # 标记为模型未识别的事件
                        })
                    else:
                        append_buffer.append({
                            "file": base,
                            "win_start": t0,
                            "win_end": t1,
                            "class": int(cid),
                            "gt_start": gt_abs,     # 绝对时间
                            "pred_start": None,  # 未识别的事件，记录为空
                            "is_recognized": False  # 标记为模型未识别的事件
                        })

    append_rows(out_csv_path, append_buffer)

# ---------------- entry ----------------
def is_sony_path(p: str) -> bool:
    return "sony" in p.lower()

def is_tau_path(p: str) -> bool:
    return "tau" in p.lower()

if __name__ == "__main__":
    files = [
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix011.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix012.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix013.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix014.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix015.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix016.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix017.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix018.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix019.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix020.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix021.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix022.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix023.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix024.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix025.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix026.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix027.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix028.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix029.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room22_mix001.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room22_mix002.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room22_mix003.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room22_mix004.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room22_mix005.wav",   
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room22_mix006.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room22_mix007.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room22_mix008.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room22_mix009.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room22_mix010.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room22_mix011.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/fold3_room6_mix006.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/fold3_room6_mix007.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/fold3_room7_mix001.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/fold3_room7_mix002.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/fold3_room7_mix003.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/fold3_room7_mix004.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/fold3_room7_mix005.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/fold3_room7_mix006.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/fold3_room9_mix001.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/fold3_room9_mix002.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/fold3_room9_mix003.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/fold3_room9_mix004.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/fold3_room9_mix005.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/fold3_room9_mix006.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/fold3_room9_mix007.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau/fold3_room9_mix008.wav",
    ]

    df_tau  = load_events_csv(TAU_EVENTS_CSV)  if os.path.exists(TAU_EVENTS_CSV)  else None
    df_sony = load_events_csv(SONY_EVENTS_CSV) if os.path.exists(SONY_EVENTS_CSV) else None

    for wav_path in files:
        for win_len in WIN_LENS:
            win_tag = f"win{int(win_len):02d}"
            replies_jsonl = f"{wav_path}.{win_tag}.4limit.aero.replies.jsonl"

            if is_tau_path(wav_path):
                if df_tau is None:
                    raise FileNotFoundError(f"TAU events CSV not found: {TAU_EVENTS_CSV}")
                out_csv = f"aero_tau.{win_tag}_earliest.aero.csv"
                process_file_windows_areo(wav_path, df_tau, out_csv, replies_jsonl, win_len)

            elif is_sony_path(wav_path):
                if df_sony is None:
                    raise FileNotFoundError(f"SONY events CSV not found: {SONY_EVENTS_CSV}")
                out_csv = f"aero_sony.{win_tag}_earliest.aero.csv"
                process_file_windows_areo(wav_path, df_sony, out_csv, replies_jsonl, win_len)

            else:
                print(f"[AERO][WARN] Unknown domain (neither sony nor tau): {wav_path}", flush=True)
