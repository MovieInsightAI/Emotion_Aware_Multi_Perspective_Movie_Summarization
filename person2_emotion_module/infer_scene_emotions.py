import pandas as pd
from tqdm import tqdm
from config import SCENE_INDEX_FILE, FEATURES_DIR, SCENE_EMOTIONS_DIR, PROJECT_EMOTIONS
from utils import read_scene_index
from emotion_classifier import AudioEmotionClassifier
from subtitle_emotion_hint import SubtitleEmotionHint


def normalize_scores(score_dict):
    total = sum(score_dict.values())
    if total == 0:
        return {k: 0.0 for k in score_dict}
    return {k: float(v / total) for k, v in score_dict.items()}


def combine_scores(audio_scores, subtitle_scores, feature_row):
    final_scores = {}

    for emotion in PROJECT_EMOTIONS:
        final_scores[emotion] = 0.75 * audio_scores.get(emotion, 0.0) + 0.25 * subtitle_scores.get(emotion, 0.0)

    tempo = feature_row.get("tempo", 0.0)
    energy = feature_row.get("rms_energy", 0.0)

    if tempo > 115:
        final_scores["tense"] += 0.08
    if energy < 0.015:
        final_scores["calm"] += 0.05
    if energy > 0.04:
        final_scores["angry"] += 0.05
        final_scores["tense"] += 0.05

    return normalize_scores(final_scores)


def infer_scene_emotions():
    scene_df = read_scene_index(SCENE_INDEX_FILE)
    features_df = pd.read_csv(FEATURES_DIR / "audio_features.csv")

    feature_lookup = {row["scene_id"]: row for _, row in features_df.iterrows()}

    audio_model = AudioEmotionClassifier()
    subtitle_model = SubtitleEmotionHint()

    rows = []

    for _, row in tqdm(scene_df.iterrows(), total=len(scene_df), desc="Inferring scene emotions"):
        scene_id = row["scene_id"]
        subtitle_text = str(row["subtitle_text"])
        feature_row = feature_lookup.get(scene_id)

        if feature_row is None:
            continue

        audio_path = feature_row["audio_path"]

        audio_scores = audio_model.predict(audio_path)
        subtitle_scores = subtitle_model.predict(subtitle_text)
        final_scores = combine_scores(audio_scores, subtitle_scores, feature_row)

        top_emotion = max(final_scores, key=final_scores.get)

        record = {
            "scene_id": scene_id,
            "audio_path": audio_path,
            "subtitle_text": subtitle_text,
            "top_emotion": top_emotion
        }

        for emotion in PROJECT_EMOTIONS:
            record[f"{emotion}_score"] = final_scores[emotion]

        rows.append(record)

    output_df = pd.DataFrame(rows)
    output_csv = SCENE_EMOTIONS_DIR / "scene_emotion_scores.csv"
    output_df.to_csv(output_csv, index=False)

    print(f"Saved final scene emotions to: {output_csv}")


if __name__ == "__main__":
    infer_scene_emotions()