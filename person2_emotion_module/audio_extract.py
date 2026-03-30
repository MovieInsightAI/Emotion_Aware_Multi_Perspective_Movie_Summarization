from pathlib import Path
from tqdm import tqdm
from config import SCENE_INDEX_FILE, AUDIO_CLIPS_DIR, SAMPLE_RATE
from utils import read_scene_index, ffmpeg_extract_audio

def extract_all_scene_audio():
    df = read_scene_index(SCENE_INDEX_FILE)
    for _, row in tqdm(df.iterrows(), total = len(df), desc = "Extracting scene audio"):
        scene_id = row["scene_id"]
        video_path = Path(row["video_path"])
        start_time = row["start_time"]
        end_time = row["end_time"]
        output_wav = AUDIO_CLIPS_DIR / f"{scene_id}.wav"

        ffmpeg_extract_audio(
            video_path=video_path,
            output_wav_path=output_wav,
            start_time=start_time,
            end_time=end_time,
            sample_rate=SAMPLE_RATE
        )
    print(f"Done. Audio clips saved in: {AUDIO_CLIPS_DIR}")
if __name__ == "__main__":
    extract_all_scene_audio()