from pathlib import Path
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from config import AUDIO_CLIPS_DIR, FEATURES_DIR, SAMPLE_RATE

def extract_features_from_audio(audio_path):
    y, sr = librosa.load(audio_path, sr = SAMPLE_RATE)
    if len(y) == 0:
        return {
            "duration": 0.0,
            "rms_energy": 0.0,
            "zero_crossing_rate": 0.0,
            "spectral_centroid": 0.0,
            "tempo": 0.0,
            "mfcc_mean": 0.0,
            "mfcc_std": 0.0
        }
    duration = librosa.get_duration(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return {
        "duration": float(duration),
        "rms_energy": float(np.mean(rms)),
        "zero_crossing_rate": float(np.mean(zcr)),
        "spectral_centroid": float(np.mean(spec_centroid)),
        "tempo": float(tempo),
        "mfcc_mean": float(np.mean(mfcc)),
        "mfcc_std": float(np.std(mfcc))
    }
def preprocess_all_audio():
    rows = []

    for audio_file in tqdm(sorted(AUDIO_CLIPS_DIR.glob("*.wav")), desc="Preprocessing audio"):
        scene_id = audio_file.stem
        features = extract_features_from_audio(audio_file)
        features["scene_id"] = scene_id
        features["audio_path"] = str(audio_file)
        rows.append(features)

    df = pd.DataFrame(rows)
    output_csv = FEATURES_DIR / "audio_features.csv"
    df.to_csv(output_csv, index=False)

    print(f"Saved features to: {output_csv}")

    if __name__ == "__main__":
        preprocess_all_audio()