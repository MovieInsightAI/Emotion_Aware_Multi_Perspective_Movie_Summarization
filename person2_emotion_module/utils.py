import os
import subprocess
from pathlib import Path
import pandas as pd


def hhmmss_to_seconds(time_str: str) -> float:
    parts = time_str.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid time format: {time_str}")
    h, m, s = parts
    return int(h) * 3600 + int(m) * 60 + float(s)


def ensure_file_exists(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")


def read_scene_index(csv_path):
    ensure_file_exists(csv_path)
    df = pd.read_csv(csv_path)
    required_cols = {"scene_id", "video_path", "start_time", "end_time", "subtitle_text"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"scene_index.csv missing columns: {missing}")
    return df


def run_command(cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed:\n{' '.join(cmd)}\n\n{result.stderr}")
    return result.stdout


def ffmpeg_extract_audio(video_path, output_wav_path, start_time, end_time, sample_rate=16000):
    ensure_file_exists(video_path)
    output_wav_path = Path(output_wav_path)
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-ss", str(start_time),
        "-to", str(end_time),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        str(output_wav_path)
    ]
    run_command(cmd)