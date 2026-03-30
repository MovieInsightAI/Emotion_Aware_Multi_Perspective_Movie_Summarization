import numpy as np
import librosa
from transformers import pipeline
from config import AUDIO_MODEL_NAME, SAMPLE_RATE, PROJECT_EMOTIONS


class AudioEmotionClassifier:
    def __init__(self):
        self.pipe = pipeline(
            task="audio-classification",
            model=AUDIO_MODEL_NAME,
            top_k=None
        )

    def _map_superb_labels_to_project_labels(self, model_scores, tempo=0.0, energy=0.0):
        mapped = {emotion: 0.0 for emotion in PROJECT_EMOTIONS}

        label_map = {
            "hap": "happy",
            "sad": "sad",
            "ang": "angry",
            "neu": "calm"
        }

        for item in model_scores:
            raw_label = item["label"].lower()
            score = float(item["score"])

            for key, target in label_map.items():
                if key in raw_label:
                    mapped[target] += score

        if mapped["angry"] > 0.35 and energy > 0.03:
            mapped["tense"] += 0.25

        if tempo > 110 and energy > 0.025:
            mapped["tense"] += 0.20

        if mapped["sad"] > 0.30 and energy < 0.02:
            mapped["fearful"] += 0.15

        total = sum(mapped.values())
        if total > 0:
            mapped = {k: v / total for k, v in mapped.items()}

        return mapped

    def predict(self, audio_path: str):
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

        if len(y) == 0:
            return {emotion: 0.0 for emotion in PROJECT_EMOTIONS}

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        energy = float(np.mean(librosa.feature.rms(y=y)[0]))

        results = self.pipe(audio_path)
        mapped_scores = self._map_superb_labels_to_project_labels(results, tempo=tempo, energy=energy)

        return mapped_scores