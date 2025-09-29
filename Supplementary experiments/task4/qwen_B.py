import os
import re
import json
import math
import soundfile as sf
import pandas as pd
import librosa
from io import BytesIO
from urllib.request import urlopen
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from typing import List, Optional

# ====================== 配置区（请按需修改） ======================

# 加载 Qwen 模型和处理器
  # 设置目标采样率

# 加载处理器并传递采样率参数
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct",
    sampling_rate=16000  # 确保传递采样率
)

# 加载模型
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "/data/user/jzt/.cache/modelscope/hub/models/Qwen/Qwen2-Audio-7B-Instruct", 
    local_files_only=True,  # 确保只从本地加载
    device_map="auto"
)
_QWEN_TARGET_SR = processor.feature_extractor.sampling_rate
# 窗口长度（秒）
WIN_LENS_ABS = [5,10,20,30,40,50,60]

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

# 要评测的音频（保证整条仅含 {1,4}）
FILES = [
    "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix001.wav",
    "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix002.wav",
]

# 输出模板
DETAIL_CSV_TPL    = "{basename}.win{win:02d}.A.detail.csv"
SUMMARY_CSV_TPL   = "{basename}.win{win:02d}.A.summary.csv"
REPLIES_JSONL_TPL = "{basename}.win{win:02d}.A.replies.jsonl"

# 类别映射：我们要分析的类别（如 "胡说" 检测的类别）
ABSENT_CANDIDATES = [0, 2, 6, 7, 8, 11, 12]

# ====================== 音频切窗 ======================

def split_fixed_win(wav_path: str, win_len: float):
    """
    返回 (wav, sr, windows)
    windows: list[(i0, i1, t0, t1)]，丢弃不足一个窗口的尾巴
    """
    wav, sr = sf.read(wav_path)
    if hasattr(wav, "ndim") and wav.ndim > 1:
        wav = wav.mean(axis=1)  # 将多声道转换为单声道
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

def load_audio_from_url(url: str, processor):
    """
    从 URL 加载音频
    """
    audio_data = librosa.load(BytesIO(urlopen(url).read()), sr=processor.feature_extractor.sampling_rate)[0]
    return audio_data

# ====================== 提示词和模型调用 ======================


def build_prompt_B(target_class: int, win_len: float) -> str:
    """
    若目标类在窗口中不存在 -> 返回 []
    若存在 -> 仅返回 [{"start": seconds}]（单对象数组，严格 JSON）
    """
    meaning = CLASS_ID_TO_NAME.get(target_class, "Unknown")
    return (
        f"You will be given a {win_len:.0f}-second audio segment.\n"
        f"Task: Detect the EARLIEST start time (in seconds) of the following target class within THIS segment:\n"
        f"{target_class}: {meaning}\n\n"
        "Return format rules (STRICT):\n"
        "1) If the target class DOES NOT occur in THIS segment, return an EMPTY JSON array: []\n"
        "2) If it DOES occur, return ONLY a strict JSON array with exactly ONE object:\n"
        "   [{\"start\": <number>}]\n"
        f"Constraints: 0 <= start < {win_len:.1f}. The start time is RELATIVE to THIS segment (0 = segment start).\n"
        "No extra keys. No extra fields. No extra text. No markdown. No code fences."
    )

def call_model_B(audio_arr, sr, target_class: int, win_len: float) -> str:
    """
    调用 Qwen 模型进行推理并获取开始时间的预测。
    """
    # 生成提示词
    text = build_prompt_B(target_class, win_len)

    # Qwen 要求的采样率为 16k
    if sr != _QWEN_TARGET_SR:
        audio_arr = librosa.resample(audio_arr, orig_sr=sr, target_sr=_QWEN_TARGET_SR)

    # 按照 ChatML 构造会话输入
    conversation = [
        {"role": "user", "content": [
            {"type": "audio", "audio_url": None},  # 本地数组，不用 URL
            {"type": "text", "text": text},
        ]},
    ]
    
    # 构建输入文本
    inputs = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

    # 将音频和文本输入一起传递给模型
    inputs = processor(text=inputs, audio=audio_arr, return_tensors="pt", padding=True)

    # 将输入数据移动到模型所在的设备
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    # 调用 Qwen 模型进行生成
    generate_ids = model.generate(**inputs, max_length=4096)

    # 只保留新生成的部分（去除输入部分）
    input_ids_length = inputs['input_ids'].size(1)  # 获取输入的长度
    generate_ids = generate_ids[:, input_ids_length:]  # 去除输入部分

    # 解析并返回模型响应
    return processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# ====================== Robust JSON 顶层数组/对象提取 ======================

import json
import re
from typing import Optional

def extract_floats_from_reply(reply: str) -> Optional[float]:
    """
    从模型的回复中提取浮点数（如 2, 2.3, 2.）。
    """
    # 正则表达式，匹配浮动数字（包括整数和小数）
    number_pattern = r"-?\d*\.\d+|-?\d+"
    
    # 查找所有匹配的数字
    numbers = re.findall(number_pattern, reply)
    
    # 如果找到了浮动数字，返回第一个匹配的数字（转换为浮动类型）
    if numbers:
        return float(numbers[0])  # 返回第一个匹配的数字
    return None

def clean_reply(reply: str) -> str:
    """
    清理并标准化 `reply` 字段中的格式，处理引号、转义字符，及缺失的括号。
    """
    # 替换单引号为双引号，并去除转义字符
    cleaned_reply = reply.replace("'", "\"")  # 将单引号替换为双引号
    cleaned_reply = re.sub(r'\\\"', '"', cleaned_reply)  # 处理转义字符

    # 检查 JSON 是否完整，如果缺少右括号，补充
    if cleaned_reply.count("{") == cleaned_reply.count("}") - 1:
        cleaned_reply += "}"  # 补充右大括号
    elif cleaned_reply.count("[") == cleaned_reply.count("]") - 1:
        cleaned_reply += "]"  # 补充右中括号

    # 如果仍然不完整，尝试补充到合法 JSON
    if not cleaned_reply.endswith("}") and not cleaned_reply.endswith("]"):
        cleaned_reply += "}"  # 默认补充大括号

    return cleaned_reply

def parse_A_reply_for_start(raw_reply: str, win_len: float) -> Optional[float]:
    """
    解析模型的回复，返回合法的开始时间（浮动数字），如果没有则返回 None。
    """
    # 清理并标准化 reply 字段中的格式
    cleaned_reply = clean_reply(raw_reply)
    
    # 使用正则提取浮动数字（start 时间）
    start_time = extract_floats_from_reply(cleaned_reply)
    
    # 确保返回的 start 时间在 [0, win_len) 范围内
    if start_time is not None and 0.0 <= start_time < win_len:
        return start_time
    return None


# ====================== 结果保存与统计 ======================

def ensure_dir_for(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def run_B_for_file(wav_path: str):
    base = os.path.basename(wav_path)
    base_noext = os.path.splitext(base)[0]

    for win_len in WIN_LENS_ABS:
        wav, sr, wins = split_fixed_win(wav_path, win_len)
        win_tag = f"win{int(win_len):02d}"

        detail_csv    = DETAIL_CSV_TPL.format(basename=base_noext, win=win_len)
        summary_csv   = SUMMARY_CSV_TPL.format(basename=base_noext, win=win_len)
        replies_jsonl = REPLIES_JSONL_TPL.format(basename=base_noext, win=win_len)

        ensure_dir_for(replies_jsonl)

        detail_rows = []
        with open(replies_jsonl, "a", encoding="utf-8") as fout:
            for (i0, i1, t0, t1) in wins:
                clip = wav[i0:i1]
                for tgt in ABSENT_CANDIDATES:
                    raw = call_model_B(clip, sr, tgt, win_len)

                    # 保存原始回复做审计
                    fout.write(json.dumps({
                        "file": base,
                        "win_start": t0,
                        "win_end": t1,
                        "target_class": int(tgt),
                        "reply": raw
                    }, ensure_ascii=False) + "\n")

                    # 解析并判断是否为胡说
                    rel = parse_A_reply_for_start(raw, win_len)
                    hallucinated = 1 if rel is not None else 0
                    abs_start = t0 + rel if rel is not None else ""

                    detail_rows.append({
                        "file": base,
                        "win_start": t0,
                        "win_end": t1,
                        "win_len": win_len,
                        "target_class": int(tgt),
                        "target_name": CLASS_ID_TO_NAME.get(int(tgt), "Unknown"),
                        "model_rel_start": rel if rel is not None else "",
                        "model_abs_start": abs_start,
                        "hallucinated": hallucinated,
                        "raw_reply": raw
                    })

        if detail_rows:
            df = pd.DataFrame(detail_rows)
            df.to_csv(detail_csv, index=False)
            grp = df.groupby(["win_len", "target_class", "target_name"], as_index=False).agg(
                n=("hallucinated", "size"),
                n_hallucinated=("hallucinated", "sum"),
            )
            grp["hallucination_rate"] = grp["n_hallucinated"] / grp["n"]
            grp.sort_values(["win_len", "target_class"], inplace=True)
            grp.to_csv(summary_csv, index=False)

            print(f"[detail] {len(df)} rows -> {detail_csv}")
            print(f"[summary] {len(grp)} groups -> {summary_csv}")
        else:
            print(f"[WARN] no windows produced for {base} @ {win_len}s")

# 入口函数
if __name__ == "__main__":
    for wav_path in FILES:
        if not os.path.exists(wav_path):
            print(f"[ERR] not found: {wav_path}")
            continue
        run_B_for_file(wav_path)

    print("\nDone. Inspect *.A.detail.csv / *.A.summary.csv / *.A.replies.jsonl\n")
